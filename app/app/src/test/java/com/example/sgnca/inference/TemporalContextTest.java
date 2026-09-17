package com.example.sgnca.inference;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

public class TemporalContextTest {
    @Test
    public void flattenUsesClassThenChronologicalFrameOrder() {
        float[][][] frameMajor = {
                {{1f, 2f}, {10f, 20f}},
                {{3f, 4f}, {30f, 40f}},
                {{5f, 6f}, {50f, 60f}}
        };

        float[] flattened = TemporalContext.flatten(frameMajor, 2, 3, 2);

        assertArrayEquals(
                new float[]{1f, 2f, 3f, 4f, 5f, 6f, 10f, 20f, 30f, 40f, 50f, 60f},
                flattened,
                0f
        );
    }
}
