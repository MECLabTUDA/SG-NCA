package com.example.sgnca;

import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.text.Spannable;
import android.text.SpannableStringBuilder;
import android.text.style.ForegroundColorSpan;
import android.text.style.StyleSpan;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.ColorUtils;

import com.example.sgnca.inference.DemoResult;
import com.example.sgnca.inference.ModelConfig;
import com.example.sgnca.inference.SemanticRelation;
import com.example.sgnca.inference.SgNcaEngine;
import com.example.sgnca.ui.SceneGraphView;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.chip.Chip;
import com.google.android.material.chip.ChipGroup;
import com.google.android.material.progressindicator.LinearProgressIndicator;

import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public final class MainActivity extends AppCompatActivity {
    private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();

    private ImageView resultImage;
    private TextView frameCounter;
    private TextView statusText;
    private TextView timingText;
    private TextView errorText;
    private SceneGraphView sceneGraphView;
    private ChipGroup objectsContainer;
    private LinearLayout relationsContainer;
    private LinearProgressIndicator progress;
    private LinearProgressIndicator contextProgress;
    private MaterialButton playButton;
    private MaterialButton previousButton;
    private MaterialButton nextButton;

    private volatile SgNcaEngine engine;
    private Future<?> activeTask;
    private Bitmap displayedBitmap;
    private int currentPosition = -1;
    private int highestAnalyzedPosition = -1;
    private volatile boolean sequenceRunning;
    private volatile boolean playbackPaused;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultImage = findViewById(R.id.result_image);
        frameCounter = findViewById(R.id.frame_counter);
        statusText = findViewById(R.id.status_text);
        timingText = findViewById(R.id.timing_text);
        errorText = findViewById(R.id.error_text);
        sceneGraphView = findViewById(R.id.scene_graph);
        objectsContainer = findViewById(R.id.objects_container);
        relationsContainer = findViewById(R.id.relations_container);
        progress = findViewById(R.id.progress);
        contextProgress = findViewById(R.id.context_progress);
        playButton = findViewById(R.id.play_button);
        previousButton = findViewById(R.id.previous_button);
        nextButton = findViewById(R.id.next_button);

        playButton.setOnClickListener(view -> togglePlayback());
        previousButton.setOnClickListener(view -> navigateTo(currentPosition - 1));
        nextButton.setOnClickListener(view -> navigateTo(currentPosition + 1));

        initializeEngine();
    }

    private void initializeEngine() {
        setBusy(true);
        statusText.setText(R.string.initializing);
        activeTask = inferenceExecutor.submit(() -> {
            try {
                SgNcaEngine initializedEngine = new SgNcaEngine(getApplicationContext());
                if (isDestroyed()) {
                    initializedEngine.close();
                    return;
                }
                engine = initializedEngine;
                runOnUiThread(() -> {
                    if (isDestroyed()) {
                        return;
                    }
                    activeTask = null;
                    progress.setMax(initializedEngine.getTargetCount());
                    setBusy(false);
                    statusText.setText(R.string.ready);
                    errorText.setVisibility(View.GONE);
                });
            } catch (Exception exception) {
                showError("Could not load the on-device models", exception);
            }
        });
    }

    private void togglePlayback() {
        if (!sequenceRunning) {
            startOrContinueSequence();
            return;
        }
        if (playbackPaused) {
            resumeSequence();
        } else {
            pauseSequence();
        }
    }

    private void startOrContinueSequence() {
        SgNcaEngine currentEngine = engine;
        if (currentEngine == null || isTaskRunning()) {
            return;
        }

        int lastPosition = currentEngine.getTargetCount() - 1;
        boolean replay = currentPosition >= lastPosition;
        sequenceRunning = true;
        playbackPaused = false;
        playButton.setText(R.string.pause);
        playButton.setIconResource(R.drawable.ic_pause);
        errorText.setVisibility(View.GONE);

        if (replay) {
            clearResults();
        }
        schedulePlaybackFrame(replay ? 0 : currentPosition + 1, replay);
    }

    private void pauseSequence() {
        playbackPaused = true;
        playButton.setText(R.string.resume);
        playButton.setIconResource(R.drawable.ic_play);
        if (isTaskRunning()) {
            statusText.setText(R.string.pause_requested);
        } else {
            showPausedStatus();
            setBusy(false);
        }
    }

    private void resumeSequence() {
        playbackPaused = false;
        playButton.setText(R.string.pause);
        playButton.setIconResource(R.drawable.ic_pause);
        if (isTaskRunning()) {
            return;
        }
        schedulePlaybackFrame(currentPosition + 1, false);
    }

    private void schedulePlaybackFrame(int position, boolean resetTimeline) {
        SgNcaEngine currentEngine = engine;
        if (currentEngine == null || position < 0 || position >= currentEngine.getTargetCount()) {
            finishSequence();
            return;
        }

        boolean cached = !resetTimeline && position <= highestAnalyzedPosition;
        int total = currentEngine.getTargetCount();
        setBusy(true);
        contextProgress.setProgressCompat(0, false);
        statusText.setText(getString(R.string.analyzing_frame, position + 1, total));
        activeTask = inferenceExecutor.submit(() -> {
            try {
                if (resetTimeline) {
                    currentEngine.resetTimeline();
                }
                DemoResult result = currentEngine.analyzeTarget(
                        position,
                        this::reportContextProgress
                );
                runOnUiThread(() -> {
                    if (isDestroyed()) {
                        return;
                    }
                    activeTask = null;
                    renderResult(result, cached);
                    if (result.position >= total - 1) {
                        finishSequence();
                    } else if (playbackPaused) {
                        showPausedStatus();
                        setBusy(false);
                    } else {
                        schedulePlaybackFrame(result.position + 1, false);
                    }
                });
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
            } catch (Exception exception) {
                showError("On-device inference failed", exception);
            }
        });
    }

    private void showPausedStatus() {
        statusText.setText(getString(
                R.string.paused_at_frame,
                currentPosition + 1,
                engine.getTargetCount()
        ));
    }

    private void finishSequence() {
        activeTask = null;
        sequenceRunning = false;
        playbackPaused = false;
        statusText.setText(R.string.sequence_complete);
        playButton.setText(R.string.replay_all);
        playButton.setIconResource(R.drawable.ic_play);
        setBusy(false);
    }

    private void reportContextProgress(int completed, int total) {
        runOnUiThread(() -> {
            if (!isDestroyed()) {
                contextProgress.setMax(total);
                contextProgress.setProgressCompat(completed, true);
            }
        });
    }

    private void navigateTo(int position) {
        SgNcaEngine currentEngine = engine;
        if (currentEngine == null
                || isTaskRunning()
                || sequenceRunning && !playbackPaused) {
            return;
        }
        if (position < 0 || position >= currentEngine.getTargetCount()) {
            return;
        }

        setBusy(true);
        errorText.setVisibility(View.GONE);
        contextProgress.setProgressCompat(0, false);
        boolean cached = position <= highestAnalyzedPosition;
        activeTask = inferenceExecutor.submit(() -> {
            try {
                DemoResult result = currentEngine.analyzeTarget(
                        position,
                        this::reportContextProgress
                );
                runOnUiThread(() -> {
                    if (isDestroyed()) {
                        return;
                    }
                    activeTask = null;
                    renderResult(result, cached);
                    if (sequenceRunning && playbackPaused) {
                        if (result.position >= currentEngine.getTargetCount() - 1) {
                            finishSequence();
                        } else {
                            showPausedStatus();
                            setBusy(false);
                        }
                    } else {
                        statusText.setText(
                                cached ? R.string.cached_result : R.string.frame_complete
                        );
                        updateIdlePlayButton();
                        setBusy(false);
                    }
                });
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
            } catch (Exception exception) {
                showError("On-device inference failed", exception);
            }
        });
    }

    private void renderResult(DemoResult result, boolean cached) {
        displayedBitmap = result.visualization;
        resultImage.setImageBitmap(displayedBitmap);
        currentPosition = result.position;
        highestAnalyzedPosition = Math.max(highestAnalyzedPosition, result.position);
        frameCounter.setText(getString(
                R.string.video_counter,
                result.position + 1,
                result.targetCount
        ));
        progress.setProgressCompat(result.position + 1, true);

        renderRelations(result);
        sceneGraphView.setGraph(
                engine.getConfig(),
                result.detectedObjects,
                result.touchingRelations,
                result.relations
        );
        renderObjectLegend(result);

        long frameTotal = 0;
        int measuredFrames = 0;
        for (long frameTime : result.frameTimesMs) {
            if (frameTime > 0) {
                frameTotal += frameTime;
                measuredFrames++;
            }
        }
        float averageFrameTime = measuredFrames == 0 ? 0 : frameTotal / (float) measuredFrames;
        if (cached) {
            timingText.setText(String.format(
                    Locale.US,
                    "Cached result · original pass %d ms",
                    result.totalTimeMs
            ));
        } else {
            timingText.setText(String.format(
                    Locale.US,
                    "%d new inputs · frame model %.0f ms avg · relations %d ms",
                    result.newlyComputedFrames,
                    averageFrameTime,
                    result.relationTimeMs
            ));
        }
    }

    private void renderRelations(DemoResult result) {
        relationsContainer.removeAllViews();
        if (result.relations.isEmpty()) {
            addContainerMessage(
                    relationsContainer,
                    getString(R.string.no_relations),
                    ContextCompat.getColor(this, R.color.mec_on_dark_muted)
            );
            return;
        }

        ModelConfig config = engine.getConfig();
        for (SemanticRelation relation : result.relations) {
            String subject = displayName(relation.subject);
            String verb = relation.verb + "s";
            String object = displayName(relation.object);
            SpannableStringBuilder line = new SpannableStringBuilder();

            int subjectStart = line.length();
            line.append(subject);
            int subjectEnd = line.length();
            line.append("  ").append(verb).append("  ");
            int objectStart = line.length();
            line.append(object);
            int objectEnd = line.length();
            int scoreStart = line.length();
            line.append(String.format(Locale.US, "   %.1f%%", relation.score * 100.0f));

            line.setSpan(
                    new ForegroundColorSpan(config.classes.get(config.classSlot(relation.subjectId)).color),
                    subjectStart,
                    subjectEnd,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
            );
            line.setSpan(
                    new StyleSpan(Typeface.BOLD),
                    subjectStart,
                    subjectEnd,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
            );
            line.setSpan(
                    new ForegroundColorSpan(config.classes.get(config.classSlot(relation.objectId)).color),
                    objectStart,
                    objectEnd,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
            );
            line.setSpan(
                    new StyleSpan(Typeface.BOLD),
                    objectStart,
                    objectEnd,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
            );
            line.setSpan(
                    new ForegroundColorSpan(ContextCompat.getColor(this, R.color.mec_on_dark_muted)),
                    scoreStart,
                    line.length(),
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
            );

            TextView row = new TextView(this);
            row.setText(line);
            row.setTextColor(ContextCompat.getColor(this, R.color.mec_on_dark));
            row.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setPadding(dp(12), dp(11), dp(12), dp(11));
            GradientDrawable background = new GradientDrawable();
            background.setColor(ContextCompat.getColor(this, R.color.mec_ink_soft));
            background.setCornerRadius(dp(12));
            row.setBackground(background);

            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
            );
            params.setMargins(0, dp(4), 0, dp(4));
            relationsContainer.addView(row, params);
        }
    }

    private void renderObjectLegend(DemoResult result) {
        objectsContainer.removeAllViews();
        if (result.detectedObjects.isEmpty()) {
            addContainerMessage(
                    objectsContainer,
                    getString(R.string.no_result),
                    ContextCompat.getColor(this, R.color.mec_text_secondary)
            );
            return;
        }

        Set<String> detected = new HashSet<>(result.detectedObjects);
        for (ModelConfig.ClassInfo classInfo : engine.getConfig().classes) {
            if (!detected.contains(classInfo.name)) {
                continue;
            }

            Chip chip = new Chip(this);
            chip.setText(displayName(classInfo.name));
            chip.setTextColor(ContextCompat.getColor(this, R.color.mec_ink));
            chip.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
            chip.setSingleLine(true);
            chip.setCheckable(false);
            chip.setClickable(false);
            chip.setEnsureMinTouchTargetSize(false);
            chip.setChipBackgroundColor(ColorStateList.valueOf(
                    ColorUtils.blendARGB(classInfo.color, Color.WHITE, 0.90f)
            ));
            chip.setChipStrokeColor(ColorStateList.valueOf(classInfo.color));
            chip.setChipStrokeWidth(dp(1));

            GradientDrawable dotBackground = new GradientDrawable();
            dotBackground.setShape(GradientDrawable.OVAL);
            dotBackground.setColor(classInfo.color);
            chip.setChipIcon(dotBackground);
            chip.setChipIconVisible(true);
            chip.setChipIconSize(dp(10));
            chip.setIconStartPadding(dp(1));
            chip.setChipStartPadding(dp(3));
            chip.setTextStartPadding(dp(5));
            chip.setTextEndPadding(dp(3));
            objectsContainer.addView(chip);
        }
    }

    private void clearResults() {
        resultImage.setImageDrawable(null);
        displayedBitmap = null;
        currentPosition = -1;
        highestAnalyzedPosition = -1;
        frameCounter.setText(R.string.video_counter_empty);
        timingText.setText("—");
        contextProgress.setProgressCompat(0, false);
        sceneGraphView.clear();
        relationsContainer.removeAllViews();
        objectsContainer.removeAllViews();
        addContainerMessage(
                relationsContainer,
                getString(R.string.no_result),
                ContextCompat.getColor(this, R.color.mec_on_dark_muted)
        );
        addContainerMessage(
                objectsContainer,
                getString(R.string.no_result),
                ContextCompat.getColor(this, R.color.mec_text_secondary)
        );
    }

    private void addContainerMessage(ViewGroup container, String text, int color) {
        TextView message = new TextView(this);
        message.setText(text);
        message.setTextColor(color);
        message.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        message.setPadding(0, dp(7), 0, dp(7));
        container.addView(message);
    }

    private static String displayName(String name) {
        return name.replace('_', ' ');
    }

    private boolean isTaskRunning() {
        return activeTask != null && !activeTask.isDone();
    }

    private void updateIdlePlayButton() {
        if (engine == null || currentPosition < 0) {
            playButton.setText(R.string.play_all);
        } else if (currentPosition >= engine.getTargetCount() - 1) {
            playButton.setText(R.string.replay_all);
        } else {
            playButton.setText(R.string.play_from_here);
        }
        playButton.setIconResource(R.drawable.ic_play);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void setBusy(boolean busy) {
        playButton.setEnabled(
                engine != null
                        && (!busy || sequenceRunning && !playbackPaused)
        );
        boolean steppingAllowed = !busy
                && engine != null
                && (!sequenceRunning || playbackPaused);
        previousButton.setEnabled(steppingAllowed && currentPosition > 0);
        nextButton.setEnabled(
                steppingAllowed
                        && currentPosition < engine.getTargetCount() - 1
        );
        progress.setIndeterminate(busy && engine == null);
    }

    private void showError(String message, Exception exception) {
        sequenceRunning = false;
        playbackPaused = false;
        runOnUiThread(() -> {
            if (isDestroyed()) {
                return;
            }
            activeTask = null;
            updateIdlePlayButton();
            setBusy(false);
            statusText.setText(message);
            errorText.setText(exception.getClass().getSimpleName() + ": " + exception.getMessage());
            errorText.setVisibility(View.VISIBLE);
        });
    }

    @Override
    protected void onDestroy() {
        playbackPaused = false;
        if (activeTask != null) {
            activeTask.cancel(true);
        }
        SgNcaEngine currentEngine = engine;
        if (currentEngine != null) {
            inferenceExecutor.submit(() -> {
                try {
                    currentEngine.close();
                } catch (Exception ignored) {
                    // The process is already tearing down; no UI remains for this error.
                }
            });
        }
        inferenceExecutor.shutdown();
        if (displayedBitmap != null) {
            resultImage.setImageDrawable(null);
            displayedBitmap = null;
        }
        super.onDestroy();
    }
}
