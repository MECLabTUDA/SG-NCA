package com.example.sgnca.inference;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.SystemClock;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Owns the two ONNX Runtime sessions and the complete offline demo pipeline. */
public final class SgNcaEngine implements AutoCloseable {
    public interface ProgressListener {
        void onFrameComplete(int completedFrames, int totalFrames);
    }

    private static final String FRAME_MODEL = "models/frame_model.onnx";
    private static final String RELATION_MODEL = "models/relation_model.onnx";
    private static final int MIN_TOUCHING_BORDER_CONTACTS = 8;

    private final AssetManager assets;
    private final ModelConfig config;
    private final DemoConfig demo;
    private final OrtEnvironment environment;
    private final OrtSession frameSession;
    private final OrtSession relationSession;
    private final RelationPostprocessor relationPostprocessor;
    private final Map<Integer, CachedFrame> frameCache = new HashMap<>();
    private final Map<Integer, DemoResult> resultCache = new HashMap<>();
    private int highestAnalyzedPosition = -1;
    private boolean closed;

    public SgNcaEngine(Context context) throws Exception {
        assets = context.getApplicationContext().getAssets();
        config = ModelConfig.load(assets);
        demo = DemoConfig.load(assets, config.temporalOffsets);
        environment = OrtEnvironment.getEnvironment("sg-nca-android");

        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            frameSession = environment.createSession(AssetUtils.readBytes(assets, FRAME_MODEL), options);
            relationSession = environment.createSession(AssetUtils.readBytes(assets, RELATION_MODEL), options);
        }
        validateSessions();
        relationPostprocessor = new RelationPostprocessor(config);
    }

    public ModelConfig getConfig() {
        return config;
    }

    public String getDemoName() {
        return demo.name;
    }

    public int getTargetCount() {
        return demo.targetFrames.size();
    }

    public int getTargetFrameNumber(int position) {
        if (position < 0 || position >= getTargetCount()) {
            throw new IndexOutOfBoundsException("Invalid demo position " + position);
        }
        return demo.targetFrames.get(position);
    }

    public synchronized DemoResult runDemo(ProgressListener progressListener) throws Exception {
        return analyzeTarget(0, progressListener);
    }

    public synchronized DemoResult analyzeTarget(
            int position,
            ProgressListener progressListener
    ) throws Exception {
        ensureOpen();
        if (position < 0 || position >= getTargetCount()) {
            throw new IndexOutOfBoundsException("Invalid demo position " + position);
        }
        DemoResult cachedResult = resultCache.get(position);
        if (cachedResult != null) {
            if (progressListener != null) {
                progressListener.onFrameComplete(config.numFrames, config.numFrames);
            }
            return cachedResult;
        }
        if (position != highestAnalyzedPosition + 1) {
            throw new IllegalStateException(
                    "New targets must be analyzed chronologically; requested " + position
            );
        }

        int targetFrame = getTargetFrameNumber(position);
        long totalStart = SystemClock.elapsedRealtimeNanos();
        float[][][] featuresByTime = new float[config.numFrames][][];
        long[] frameTimesMs = new long[config.numFrames];
        FrameOutput targetOutput = null;
        Bitmap targetBitmap = null;
        int newlyComputedFrames = 0;

        try {
            for (int contextIndex = 0; contextIndex < config.numFrames; contextIndex++) {
                if (Thread.currentThread().isInterrupted()) {
                    throw new InterruptedException("Demo inference was cancelled");
                }
                int sourceFrame = targetFrame + config.temporalOffsets[contextIndex];
                CachedFrame cachedFrame = frameCache.get(sourceFrame);
                if (cachedFrame == null) {
                    Bitmap bitmap = loadBitmap(demo.framePath(sourceFrame));
                    try {
                        long frameStart = SystemClock.elapsedRealtimeNanos();
                        FrameOutput output = runFrame(bitmap);
                        frameTimesMs[contextIndex] = elapsedMillis(frameStart);
                        newlyComputedFrames++;
                        cachedFrame = new CachedFrame(output.objectFeatures);
                        frameCache.put(sourceFrame, cachedFrame);

                        if (sourceFrame == targetFrame) {
                            targetBitmap = bitmap;
                            targetOutput = output;
                            bitmap = null;
                        }
                    } finally {
                        if (bitmap != null && !bitmap.isRecycled()) {
                            bitmap.recycle();
                        }
                    }
                }
                featuresByTime[contextIndex] = cachedFrame.objectFeatures;
                if (progressListener != null) {
                    progressListener.onFrameComplete(contextIndex + 1, config.numFrames);
                }
            }

            if (targetBitmap == null || targetOutput == null) {
                throw new IllegalStateException("Target frame was unexpectedly precomputed");
            }

            long relationStart = SystemClock.elapsedRealtimeNanos();
            float[][] relationLogits = runRelationModel(featuresByTime);
            List<SemanticRelation> relations = relationPostprocessor.process(
                    relationLogits,
                    targetOutput.presence,
                    targetFrame
            );
            long relationTimeMs = elapsedMillis(relationStart);

            Visualization visualization = createVisualization(
                    targetBitmap,
                    targetOutput.segmentationLogits,
                    targetOutput.presence
            );
            List<String> detectedObjects = new ArrayList<>();
            for (int index = 0; index < targetOutput.presence.length; index++) {
                if (targetOutput.presence[index]) {
                    detectedObjects.add(config.classes.get(index).name);
                }
            }

            DemoResult result = new DemoResult(
                    visualization.bitmap,
                    visualization.overlayPixelCount,
                    detectedObjects,
                    relations,
                    visualization.touchingRelations,
                    frameTimesMs,
                    relationTimeMs,
                    elapsedMillis(totalStart),
                    targetFrame,
                    position,
                    getTargetCount(),
                    newlyComputedFrames
            );
            resultCache.put(position, result);
            highestAnalyzedPosition = position;
            return result;
        } finally {
            if (targetBitmap != null && !targetBitmap.isRecycled()) {
                targetBitmap.recycle();
            }
        }
    }

    public synchronized void resetTimeline() {
        for (DemoResult result : resultCache.values()) {
            if (!result.visualization.isRecycled()) {
                result.visualization.recycle();
            }
        }
        resultCache.clear();
        frameCache.clear();
        relationPostprocessor.reset();
        highestAnalyzedPosition = -1;
    }

    private FrameOutput runFrame(Bitmap bitmap) throws OrtException {
        float[] inputValues = bitmapToNchw(bitmap);
        try (OnnxTensor input = OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(inputValues),
                new long[]{1, 3, config.imageSize, config.imageSize}
        ); OrtSession.Result result = frameSession.run(Collections.singletonMap("image", input))) {
            float[][][][] logits = (float[][][][]) requiredOutput(result, "segmentation_logits").getValue();
            float[][][] features = (float[][][]) requiredOutput(result, "object_features").getValue();
            boolean[][] presence = (boolean[][]) requiredOutput(result, "presence").getValue();
            return new FrameOutput(logits[0], features[0], presence[0]);
        }
    }

    private float[][] runRelationModel(float[][][] featuresByTime) throws OrtException {
        float[] flattenedContext = TemporalContext.flatten(
                featuresByTime,
                config.classes.size(),
                config.numFrames,
                config.featureSize
        );

        try (OnnxTensor input = OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(flattenedContext),
                new long[]{config.classes.size(), config.numFrames, config.featureSize}
        ); OrtSession.Result result = relationSession.run(
                Collections.singletonMap("object_context", input)
        )) {
            return (float[][]) requiredOutput(result, "relation_logits").getValue();
        }
    }

    private static OnnxValue requiredOutput(OrtSession.Result result, String name) {
        return result.get(name).orElseThrow(
                () -> new IllegalStateException("Model output is missing: " + name)
        );
    }

    private Bitmap loadBitmap(String path) throws IOException {
        try (InputStream input = assets.open(path)) {
            Bitmap decoded = BitmapFactory.decodeStream(input);
            if (decoded == null) {
                throw new IOException("Unable to decode " + path);
            }
            if (decoded.getWidth() == config.imageSize && decoded.getHeight() == config.imageSize) {
                return decoded.copy(Bitmap.Config.ARGB_8888, false);
            }
            Bitmap scaled = Bitmap.createScaledBitmap(decoded, config.imageSize, config.imageSize, true);
            decoded.recycle();
            return scaled.copy(Bitmap.Config.ARGB_8888, false);
        }
    }

    private float[] bitmapToNchw(Bitmap bitmap) {
        int pixelCount = config.imageSize * config.imageSize;
        int[] pixels = new int[pixelCount];
        bitmap.getPixels(pixels, 0, config.imageSize, 0, 0, config.imageSize, config.imageSize);
        float[] values = new float[pixelCount * 3];
        for (int index = 0; index < pixelCount; index++) {
            int pixel = pixels[index];
            values[index] = ((pixel >> 16) & 0xFF) / 255.0f;
            values[pixelCount + index] = ((pixel >> 8) & 0xFF) / 255.0f;
            values[2 * pixelCount + index] = (pixel & 0xFF) / 255.0f;
        }
        return values;
    }

    private Visualization createVisualization(
            Bitmap source,
            float[][][] logits,
            boolean[] presence
    ) {
        int width = config.imageSize;
        int height = config.imageSize;
        int[] sourcePixels = new int[width * height];
        int[] outputPixels = new int[width * height];
        int[] winningClasses = new int[width * height];
        Arrays.fill(winningClasses, -1);
        source.getPixels(sourcePixels, 0, width, 0, 0, width, height);
        int overlayCount = 0;
        final float overlayAlpha = 0.48f;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixelIndex = y * width + x;
                int winningClass = -1;
                float winningProbability = config.maskThreshold;
                for (int classIndex = 0; classIndex < config.classes.size(); classIndex++) {
                    float probability = RelationPostprocessor.sigmoid(logits[classIndex][y][x]);
                    if (probability > winningProbability) {
                        winningProbability = probability;
                        winningClass = classIndex;
                    }
                }

                int sourceColor = sourcePixels[pixelIndex];
                if (winningClass < 0) {
                    outputPixels[pixelIndex] = sourceColor;
                    continue;
                }
                overlayCount++;
                winningClasses[pixelIndex] = winningClass;
                int overlayColor = config.classes.get(winningClass).color;
                int red = blend((sourceColor >> 16) & 0xFF, (overlayColor >> 16) & 0xFF, overlayAlpha);
                int green = blend((sourceColor >> 8) & 0xFF, (overlayColor >> 8) & 0xFF, overlayAlpha);
                int blue = blend(sourceColor & 0xFF, overlayColor & 0xFF, overlayAlpha);
                outputPixels[pixelIndex] = 0xFF000000 | (red << 16) | (green << 8) | blue;
            }
        }

        Bitmap visualization = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        visualization.setPixels(outputPixels, 0, width, 0, 0, width, height);
        return new Visualization(
                visualization,
                overlayCount,
                findTouchingRelations(winningClasses, presence, width, height)
        );
    }

    private List<TouchingRelation> findTouchingRelations(
            int[] winningClasses,
            boolean[] presence,
            int width,
            int height
    ) {
        int classCount = config.classes.size();
        int[][] contacts = new int[classCount][classCount];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixelIndex = y * width + x;
                int sourceClass = winningClasses[pixelIndex];
                if (sourceClass < 0) {
                    continue;
                }
                if (x + 1 < width) {
                    countBorderContact(sourceClass, winningClasses[pixelIndex + 1], contacts);
                }
                if (y + 1 < height) {
                    countBorderContact(sourceClass, winningClasses[pixelIndex + width], contacts);
                    if (x + 1 < width) {
                        countBorderContact(
                                sourceClass,
                                winningClasses[pixelIndex + width + 1],
                                contacts
                        );
                    }
                    if (x > 0) {
                        countBorderContact(
                                sourceClass,
                                winningClasses[pixelIndex + width - 1],
                                contacts
                        );
                    }
                }
            }
        }

        List<TouchingRelation> touchingRelations = new ArrayList<>();
        for (int first = 0; first < classCount; first++) {
            if (!presence[first]) {
                continue;
            }
            for (int second = first + 1; second < classCount; second++) {
                int borderContacts = contacts[first][second];
                if (!presence[second] || borderContacts < MIN_TOUCHING_BORDER_CONTACTS) {
                    continue;
                }
                ModelConfig.ClassInfo firstClass = config.classes.get(first);
                ModelConfig.ClassInfo secondClass = config.classes.get(second);
                touchingRelations.add(new TouchingRelation(
                        firstClass.id,
                        secondClass.id,
                        firstClass.name,
                        secondClass.name,
                        borderContacts
                ));
            }
        }
        touchingRelations.sort(
                (first, second) -> Integer.compare(second.borderContacts, first.borderContacts)
        );
        return touchingRelations;
    }

    private static void countBorderContact(int first, int second, int[][] contacts) {
        if (first < 0 || second < 0 || first == second) {
            return;
        }
        int lower = Math.min(first, second);
        int upper = Math.max(first, second);
        contacts[lower][upper]++;
    }

    private static int blend(int source, int overlay, float alpha) {
        return Math.round(source * (1.0f - alpha) + overlay * alpha);
    }

    private void validateSessions() throws OrtException {
        requireShape(frameSession.getInputInfo(), "image", 1, 3, 256, 256);
        requireShape(frameSession.getOutputInfo(), "segmentation_logits", 1, 10, 256, 256);
        requireShape(frameSession.getOutputInfo(), "object_features", 1, 10, 165);
        requireShape(frameSession.getOutputInfo(), "presence", 1, 10);
        requireShape(relationSession.getInputInfo(), "object_context", 10, 8, 165);
        requireShape(relationSession.getOutputInfo(), "relation_logits", 14, 9);
    }

    private static void requireShape(Map<String, NodeInfo> nodes, String name, long... expected) {
        NodeInfo node = nodes.get(name);
        if (node == null || !(node.getInfo() instanceof TensorInfo)) {
            throw new IllegalStateException("Missing tensor " + name);
        }
        long[] actual = ((TensorInfo) node.getInfo()).getShape();
        if (!Arrays.equals(expected, actual)) {
            throw new IllegalStateException(
                    "Unexpected shape for " + name + ": " + Arrays.toString(actual)
            );
        }
    }

    private static long elapsedMillis(long startNanos) {
        return Math.max(1L, (SystemClock.elapsedRealtimeNanos() - startNanos) / 1_000_000L);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Inference engine is closed");
        }
    }

    @Override
    public synchronized void close() throws OrtException {
        if (!closed) {
            closed = true;
            resetTimeline();
            try {
                relationSession.close();
            } finally {
                frameSession.close();
            }
        }
    }

    private static final class FrameOutput {
        final float[][][] segmentationLogits;
        final float[][] objectFeatures;
        final boolean[] presence;

        FrameOutput(float[][][] segmentationLogits, float[][] objectFeatures, boolean[] presence) {
            this.segmentationLogits = segmentationLogits;
            this.objectFeatures = objectFeatures;
            this.presence = presence;
        }
    }

    private static final class CachedFrame {
        final float[][] objectFeatures;

        CachedFrame(float[][] objectFeatures) {
            this.objectFeatures = objectFeatures;
        }
    }

    private static final class Visualization {
        final Bitmap bitmap;
        final int overlayPixelCount;
        final List<TouchingRelation> touchingRelations;

        Visualization(
                Bitmap bitmap,
                int overlayPixelCount,
                List<TouchingRelation> touchingRelations
        ) {
            this.bitmap = bitmap;
            this.overlayPixelCount = overlayPixelCount;
            this.touchingRelations = touchingRelations;
        }
    }
}
