package com.example.sgnca.inference;

import android.content.res.AssetManager;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class ModelConfig {
    public static final class ClassInfo {
        public final int id;
        public final String name;
        public final int color;

        ClassInfo(int id, String name, int color) {
            this.id = id;
            this.name = name;
            this.color = color;
        }
    }

    public static final class Pair {
        public final int subjectId;
        public final int objectId;

        public Pair(int subjectId, int objectId) {
            this.subjectId = subjectId;
            this.objectId = objectId;
        }
    }

    public final int imageSize;
    public final int numFrames;
    public final int featureSize;
    public final float maskThreshold;
    public final float relationThreshold;
    public final int smoothingWindow;
    public final List<ClassInfo> classes;
    public final List<String> verbs;
    public final List<Pair> possiblePairs;
    public final int[] temporalOffsets;
    private final Map<Integer, Integer> classSlots;

    ModelConfig(
            int imageSize,
            int numFrames,
            int featureSize,
            float maskThreshold,
            float relationThreshold,
            int smoothingWindow,
            List<ClassInfo> classes,
            List<String> verbs,
            List<Pair> possiblePairs,
            int[] temporalOffsets
    ) {
        this.imageSize = imageSize;
        this.numFrames = numFrames;
        this.featureSize = featureSize;
        this.maskThreshold = maskThreshold;
        this.relationThreshold = relationThreshold;
        this.smoothingWindow = smoothingWindow;
        this.classes = Collections.unmodifiableList(new ArrayList<>(classes));
        this.verbs = Collections.unmodifiableList(new ArrayList<>(verbs));
        this.possiblePairs = Collections.unmodifiableList(new ArrayList<>(possiblePairs));
        this.temporalOffsets = temporalOffsets.clone();
        this.classSlots = new HashMap<>();
        for (int index = 0; index < classes.size(); index++) {
            classSlots.put(classes.get(index).id, index);
        }
    }

    public int classSlot(int classId) {
        Integer slot = classSlots.get(classId);
        if (slot == null) {
            throw new IllegalArgumentException("Unknown class id " + classId);
        }
        return slot;
    }

    public String className(int classId) {
        return classes.get(classSlot(classId)).name;
    }

    public static ModelConfig load(AssetManager assets) throws IOException, JSONException {
        JSONObject root = new JSONObject(AssetUtils.readText(assets, "models/model_config.json"));
        if (root.getInt("schema_version") != 1) {
            throw new JSONException("Unsupported model metadata schema");
        }

        JSONArray inputShape = root.getJSONObject("input").getJSONArray("shape");
        int imageSize = inputShape.getInt(2);
        JSONArray relationShape = root.getJSONObject("relation_input").getJSONArray("shape");
        int featureSize = relationShape.getInt(2);

        List<ClassInfo> classes = new ArrayList<>();
        JSONArray classArray = root.getJSONArray("classes");
        for (int index = 0; index < classArray.length(); index++) {
            JSONObject item = classArray.getJSONObject(index);
            JSONArray color = item.getJSONArray("color");
            int packedColor = 0xFF000000
                    | (color.getInt(0) << 16)
                    | (color.getInt(1) << 8)
                    | color.getInt(2);
            classes.add(new ClassInfo(item.getInt("id"), item.getString("name"), packedColor));
        }

        List<String> verbs = new ArrayList<>();
        JSONArray verbArray = root.getJSONArray("verbs");
        for (int index = 0; index < verbArray.length(); index++) {
            verbs.add(verbArray.getString(index));
        }

        List<Pair> pairs = new ArrayList<>();
        JSONArray pairArray = root.getJSONArray("possible_pairs");
        for (int index = 0; index < pairArray.length(); index++) {
            JSONArray pair = pairArray.getJSONArray(index);
            pairs.add(new Pair(pair.getInt(0), pair.getInt(1)));
        }

        JSONArray offsetArray = root.getJSONArray("temporal_offsets");
        int[] offsets = new int[offsetArray.length()];
        for (int index = 0; index < offsetArray.length(); index++) {
            offsets[index] = offsetArray.getInt(index);
        }

        ModelConfig config = new ModelConfig(
                imageSize,
                root.getInt("num_frames"),
                featureSize,
                (float) root.getDouble("mask_threshold"),
                (float) root.getDouble("relation_threshold"),
                root.getInt("smoothing_window"),
                classes,
                verbs,
                pairs,
                offsets
        );
        config.validate();
        return config;
    }

    private void validate() throws JSONException {
        if (imageSize != 256 || classes.size() != 10 || numFrames != 8 || featureSize != 165) {
            throw new JSONException("Unexpected frame model contract");
        }
        if (possiblePairs.size() != 14 || verbs.size() != 9 || temporalOffsets.length != numFrames) {
            throw new JSONException("Unexpected relation model contract");
        }
    }
}
