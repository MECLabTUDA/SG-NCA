package com.example.sgnca;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import android.os.SystemClock;
import android.widget.TextView;

import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import com.google.android.material.button.MaterialButton;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.concurrent.atomic.AtomicBoolean;

@RunWith(AndroidJUnit4.class)
public class MainActivityNavigationTest {
    private static final long TIMEOUT_MS = 30_000L;

    @Test
    public void playRendersSequenceAndManualButtonsMoveOneTarget() {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            waitUntil(scenario, activity -> button(activity, R.id.play_button).isEnabled());

            AtomicBoolean sawIntermediateFrame = new AtomicBoolean();
            scenario.onActivity(activity -> button(activity, R.id.play_button).performClick());
            waitUntil(scenario, activity -> {
                String counter = text(activity, R.id.frame_counter);
                if (!counter.contains("—/20") && !counter.contains("20/20")) {
                    sawIntermediateFrame.set(true);
                }
                return counter.contains("20/20")
                        && text(activity, R.id.play_button).equals("Replay all");
            });
            assertTrue(sawIntermediateFrame.get());
            scenario.onActivity(activity -> {
                assertTrue(button(activity, R.id.previous_button).isEnabled());
                assertFalse(button(activity, R.id.next_button).isEnabled());
                assertTrue(
                        activity.findViewById(R.id.relations_heading).getTop()
                                < activity.findViewById(R.id.legend_heading).getTop()
                );
                button(activity, R.id.previous_button).performClick();
            });

            waitForCounter(scenario, "Cholec VID01 · 19/20");
            scenario.onActivity(activity -> {
                assertTrue(button(activity, R.id.previous_button).isEnabled());
                assertTrue(button(activity, R.id.next_button).isEnabled());
                button(activity, R.id.next_button).performClick();
            });

            waitForCounter(scenario, "Cholec VID01 · 20/20");
        }
    }

    private static MaterialButton button(MainActivity activity, int id) {
        return activity.findViewById(id);
    }

    private static String text(MainActivity activity, int id) {
        TextView view = activity.findViewById(id);
        return view.getText().toString();
    }

    private static void waitForCounter(
            ActivityScenario<MainActivity> scenario,
            String expected
    ) {
        waitUntil(scenario, activity -> text(activity, R.id.frame_counter).equals(expected));
    }

    private static void waitUntil(
            ActivityScenario<MainActivity> scenario,
            ActivityCondition condition
    ) {
        long deadline = SystemClock.elapsedRealtime() + TIMEOUT_MS;
        AtomicBoolean satisfied = new AtomicBoolean();
        while (SystemClock.elapsedRealtime() < deadline) {
            scenario.onActivity(activity -> satisfied.set(condition.matches(activity)));
            if (satisfied.get()) {
                return;
            }
            SystemClock.sleep(100L);
        }
        fail("Timed out waiting for the expected activity state");
    }

    private interface ActivityCondition {
        boolean matches(MainActivity activity);
    }
}
