package com.example.sgnca.inference;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.Map;

public class TemporalSmootherTest {
    @Test
    public void averagesOnlyFramesWherePairExistsAndEvictsOldFrames() {
        TemporalSmoother smoother = new TemporalSmoother(2);
        long pair = TemporalSmoother.pairKey(5, 10);

        Map<Long, float[]> first = new LinkedHashMap<>();
        first.put(pair, new float[]{0.2f, 0.8f});
        smoother.update(first);

        Map<Long, float[]> second = new LinkedHashMap<>();
        second.put(pair, new float[]{0.6f, 0.4f});
        Map<Long, float[]> averaged = smoother.update(second);
        assertArrayEquals(new float[]{0.4f, 0.6f}, averaged.get(pair), 1e-6f);

        Map<Long, float[]> absent = smoother.update(new LinkedHashMap<>());
        assertArrayEquals(new float[]{0.6f, 0.4f}, absent.get(pair), 1e-6f);
        assertEquals(2, smoother.size());
    }

    @Test
    public void pairKeyRoundTrips() {
        long key = TemporalSmoother.pairKey(9, 1);
        assertEquals(9, TemporalSmoother.subjectFromKey(key));
        assertEquals(1, TemporalSmoother.objectFromKey(key));
    }

    @Test
    public void explicitFrameIndicesPreserveAOneSecondSourceWindow() {
        TemporalSmoother smoother = new TemporalSmoother(25);
        long pair = TemporalSmoother.pairKey(5, 10);
        Map<Long, float[]> values = new LinkedHashMap<>();
        values.put(pair, new float[]{0.5f});

        smoother.update(500, values);
        smoother.update(507, values);
        smoother.update(514, values);
        smoother.update(521, values);
        assertEquals(4, smoother.size());

        smoother.update(528, values);
        assertEquals(4, smoother.size());
    }
}
