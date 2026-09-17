package com.example.sgnca;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import android.content.Context;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.example.sgnca.inference.DemoResult;
import com.example.sgnca.inference.SemanticRelation;
import com.example.sgnca.inference.SgNcaEngine;

import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public class SgNcaInstrumentedTest {
    @Test
    public void bundledTimelineUsesExactCachedContextsAndSupportsNavigation() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        try (SgNcaEngine engine = new SgNcaEngine(context)) {
            DemoResult first = engine.runDemo(null);
            assertEquals(20, engine.getTargetCount());
            assertEquals(500, first.targetFrame);
            assertEquals(8, first.newlyComputedFrames);
            assertTrue(first.overlayPixelCount > 0);
            assertFalse(first.detectedObjects.isEmpty());
            assertFalse(first.relations.isEmpty());

            SemanticRelation strongest = first.relations.get(0);
            assertEquals("grasper", strongest.subject);
            assertEquals("retract", strongest.verb);
            assertEquals("gallbladder", strongest.object);
            assertEquals(0.938f, strongest.score, 0.02f);

            DemoResult second = engine.analyzeTarget(1, null);
            assertEquals(507, second.targetFrame);
            assertEquals(2, second.newlyComputedFrames);
            assertTrue(second.overlayPixelCount > 0);
            assertFalse(second.relations.isEmpty());

            DemoResult previous = engine.analyzeTarget(0, null);
            assertTrue(first == previous);

            for (int position = 2; position < engine.getTargetCount(); position++) {
                DemoResult result = engine.analyzeTarget(position, null);
                assertEquals(engine.getTargetFrameNumber(position), result.targetFrame);
                assertEquals(2, result.newlyComputedFrames);
                assertTrue(result.overlayPixelCount > 0);
            }

            engine.resetTimeline();
            DemoResult restarted = engine.runDemo(null);
            assertEquals(8, restarted.newlyComputedFrames);
            assertEquals(500, restarted.targetFrame);
        }
    }
}
