package com.example.sgnca.inference;

import android.graphics.Bitmap;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class DemoResult {
    public final Bitmap visualization;
    public final int overlayPixelCount;
    public final List<String> detectedObjects;
    public final List<SemanticRelation> relations;
    public final List<TouchingRelation> touchingRelations;
    public final long[] frameTimesMs;
    public final long relationTimeMs;
    public final long totalTimeMs;
    public final int targetFrame;
    public final int position;
    public final int targetCount;
    public final int newlyComputedFrames;

    DemoResult(
            Bitmap visualization,
            int overlayPixelCount,
            List<String> detectedObjects,
            List<SemanticRelation> relations,
            List<TouchingRelation> touchingRelations,
            long[] frameTimesMs,
            long relationTimeMs,
            long totalTimeMs,
            int targetFrame,
            int position,
            int targetCount,
            int newlyComputedFrames
    ) {
        this.visualization = visualization;
        this.overlayPixelCount = overlayPixelCount;
        this.detectedObjects = Collections.unmodifiableList(new ArrayList<>(detectedObjects));
        this.relations = Collections.unmodifiableList(new ArrayList<>(relations));
        this.touchingRelations = Collections.unmodifiableList(new ArrayList<>(touchingRelations));
        this.frameTimesMs = frameTimesMs.clone();
        this.relationTimeMs = relationTimeMs;
        this.totalTimeMs = totalTimeMs;
        this.targetFrame = targetFrame;
        this.position = position;
        this.targetCount = targetCount;
        this.newlyComputedFrames = newlyComputedFrames;
    }
}
