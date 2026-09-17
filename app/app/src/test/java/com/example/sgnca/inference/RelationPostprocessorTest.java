package com.example.sgnca.inference;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class RelationPostprocessorTest {
    private static ModelConfig config() {
        List<ModelConfig.ClassInfo> classes = Arrays.asList(
                new ModelConfig.ClassInfo(5, "grasper", 0xFFFF0000),
                new ModelConfig.ClassInfo(10, "gallbladder", 0xFF00FF00)
        );
        return new ModelConfig(
                256,
                8,
                165,
                0.5f,
                0.5f,
                25,
                classes,
                Arrays.asList("grasp", "retract"),
                Collections.singletonList(new ModelConfig.Pair(5, 10)),
                new int[]{-25, -21, -18, -14, -11, -7, -4, 0}
        );
    }

    @Test
    public void filtersAbsentPairsAndSortsScores() {
        RelationPostprocessor postprocessor = new RelationPostprocessor(config());
        float[][] logits = {{1.3862944f, 2.1972246f}}; // sigmoid = 0.8, 0.9

        List<SemanticRelation> relations = postprocessor.process(logits, new boolean[]{true, true});

        assertEquals(2, relations.size());
        assertEquals("retract", relations.get(0).verb);
        assertEquals("grasper retracts gallbladder", relations.get(0).sentence());
        assertEquals(0.9f, relations.get(0).score, 1e-5f);

        postprocessor.reset();
        assertTrue(postprocessor.process(logits, new boolean[]{true, false}).isEmpty());
    }

    @Test
    public void sigmoidIsStableForLargeValues() {
        assertEquals(1.0f, RelationPostprocessor.sigmoid(100f), 0f);
        assertEquals(0.0f, RelationPostprocessor.sigmoid(-100f), 1e-40f);
        assertEquals(0.5f, RelationPostprocessor.sigmoid(0f), 0f);
    }
}
