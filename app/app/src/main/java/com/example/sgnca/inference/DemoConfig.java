package com.example.sgnca.inference;

import android.content.res.AssetManager;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

final class DemoConfig {
    final String name;
    final List<Integer> targetFrames;
    private final Set<Integer> availableFrames;

    private DemoConfig(String name, List<Integer> targetFrames, Set<Integer> availableFrames) {
        this.name = name;
        this.targetFrames = Collections.unmodifiableList(targetFrames);
        this.availableFrames = Collections.unmodifiableSet(availableFrames);
    }

    static DemoConfig load(AssetManager assets, int[] expectedTemporalOffsets)
            throws IOException, JSONException {
        JSONObject root = new JSONObject(AssetUtils.readText(assets, "demo/demo_config.json"));
        JSONArray offsets = root.getJSONArray("temporal_offsets");
        if (offsets.length() != expectedTemporalOffsets.length) {
            throw new JSONException("Demo temporal offsets do not match the model");
        }
        for (int index = 0; index < offsets.length(); index++) {
            if (offsets.getInt(index) != expectedTemporalOffsets[index]) {
                throw new JSONException("Demo temporal offsets do not match the model");
            }
        }

        List<Integer> targetFrames = new ArrayList<>();
        JSONArray targets = root.getJSONArray("target_frames");
        for (int index = 0; index < targets.length(); index++) {
            targetFrames.add(targets.getInt(index));
        }
        Set<Integer> availableFrames = new HashSet<>();
        JSONArray available = root.getJSONArray("available_frames");
        for (int index = 0; index < available.length(); index++) {
            availableFrames.add(available.getInt(index));
        }

        for (int target : targetFrames) {
            for (int offset : expectedTemporalOffsets) {
                if (!availableFrames.contains(target + offset)) {
                    throw new JSONException("Missing context frame " + (target + offset));
                }
            }
        }
        if (targetFrames.isEmpty()) {
            throw new JSONException("The demo has no target frames");
        }
        return new DemoConfig(root.getString("name"), targetFrames, availableFrames);
    }

    String framePath(int frameNumber) {
        if (!availableFrames.contains(frameNumber)) {
            throw new IllegalArgumentException("Frame is not bundled: " + frameNumber);
        }
        return String.format(java.util.Locale.US, "demo/frames/%08d.png", frameNumber);
    }
}
