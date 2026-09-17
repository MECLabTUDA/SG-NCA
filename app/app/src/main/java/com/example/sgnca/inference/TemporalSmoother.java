package com.example.sgnca.inference;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.Map;

/** A pair-keyed moving average equivalent to vid_inference/temporal_smoother.py. */
public final class TemporalSmoother {
    private final int windowSize;
    private final Deque<FrameProbabilities> history = new ArrayDeque<>();
    private int nextImplicitFrame;

    public TemporalSmoother(int windowSize) {
        if (windowSize < 1) {
            throw new IllegalArgumentException("windowSize must be positive");
        }
        this.windowSize = windowSize;
    }

    public synchronized Map<Long, float[]> update(Map<Long, float[]> current) {
        return update(nextImplicitFrame++, current);
    }

    public synchronized Map<Long, float[]> update(int frameIndex, Map<Long, float[]> current) {
        if (!history.isEmpty() && frameIndex <= history.getLast().frameIndex) {
            throw new IllegalArgumentException("Frame indices must be strictly increasing");
        }
        nextImplicitFrame = Math.max(nextImplicitFrame, frameIndex + 1);
        history.addLast(new FrameProbabilities(frameIndex, deepCopy(current)));
        while (!history.isEmpty() && frameIndex - history.getFirst().frameIndex >= windowSize) {
            history.removeFirst();
        }

        Map<Long, float[]> sums = new LinkedHashMap<>();
        Map<Long, Integer> counts = new LinkedHashMap<>();
        for (FrameProbabilities frame : history) {
            for (Map.Entry<Long, float[]> entry : frame.values.entrySet()) {
                float[] sum = sums.computeIfAbsent(entry.getKey(), key -> new float[entry.getValue().length]);
                float[] values = entry.getValue();
                for (int index = 0; index < values.length; index++) {
                    sum[index] += values[index];
                }
                counts.put(entry.getKey(), counts.getOrDefault(entry.getKey(), 0) + 1);
            }
        }

        Map<Long, float[]> averaged = new LinkedHashMap<>();
        for (Map.Entry<Long, float[]> entry : sums.entrySet()) {
            float[] values = entry.getValue();
            int count = counts.get(entry.getKey());
            for (int index = 0; index < values.length; index++) {
                values[index] /= count;
            }
            averaged.put(entry.getKey(), values);
        }
        return averaged;
    }

    public synchronized void reset() {
        history.clear();
        nextImplicitFrame = 0;
    }

    public synchronized int size() {
        return history.size();
    }

    public static long pairKey(int subjectId, int objectId) {
        return ((long) subjectId << 32) | (objectId & 0xffffffffL);
    }

    public static int subjectFromKey(long key) {
        return (int) (key >> 32);
    }

    public static int objectFromKey(long key) {
        return (int) key;
    }

    private static Map<Long, float[]> deepCopy(Map<Long, float[]> values) {
        Map<Long, float[]> copy = new LinkedHashMap<>();
        for (Map.Entry<Long, float[]> entry : values.entrySet()) {
            copy.put(entry.getKey(), entry.getValue().clone());
        }
        return copy;
    }

    private static final class FrameProbabilities {
        final int frameIndex;
        final Map<Long, float[]> values;

        FrameProbabilities(int frameIndex, Map<Long, float[]> values) {
            this.frameIndex = frameIndex;
            this.values = values;
        }
    }
}
