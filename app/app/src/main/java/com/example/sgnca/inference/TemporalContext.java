package com.example.sgnca.inference;

/** Converts frame-major object features into the model's class-major context tensor. */
public final class TemporalContext {
    private TemporalContext() {}

    public static float[] flatten(
            float[][][] featuresByTime,
            int classCount,
            int frameCount,
            int featureSize
    ) {
        if (featuresByTime.length != frameCount) {
            throw new IllegalArgumentException("Unexpected temporal frame count");
        }
        float[] flattened = new float[classCount * frameCount * featureSize];
        int destination = 0;
        for (int classIndex = 0; classIndex < classCount; classIndex++) {
            for (int frameIndex = 0; frameIndex < frameCount; frameIndex++) {
                if (featuresByTime[frameIndex].length != classCount
                        || featuresByTime[frameIndex][classIndex].length != featureSize) {
                    throw new IllegalArgumentException("Unexpected object feature shape");
                }
                System.arraycopy(
                        featuresByTime[frameIndex][classIndex],
                        0,
                        flattened,
                        destination,
                        featureSize
                );
                destination += featureSize;
            }
        }
        return flattened;
    }
}
