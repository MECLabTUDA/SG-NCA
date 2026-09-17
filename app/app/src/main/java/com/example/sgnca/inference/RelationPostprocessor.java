package com.example.sgnca.inference;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class RelationPostprocessor {
    private final ModelConfig config;
    private final TemporalSmoother smoother;

    public RelationPostprocessor(ModelConfig config) {
        this(config, new TemporalSmoother(config.smoothingWindow));
    }

    RelationPostprocessor(ModelConfig config, TemporalSmoother smoother) {
        this.config = config;
        this.smoother = smoother;
    }

    public List<SemanticRelation> process(float[][] logits, boolean[] presence) {
        return buildRelations(smoother.update(currentProbabilities(logits, presence)));
    }

    public List<SemanticRelation> process(float[][] logits, boolean[] presence, int frameIndex) {
        return buildRelations(smoother.update(frameIndex, currentProbabilities(logits, presence)));
    }

    private Map<Long, float[]> currentProbabilities(float[][] logits, boolean[] presence) {
        if (logits.length != config.possiblePairs.size()) {
            throw new IllegalArgumentException("Unexpected relation row count");
        }
        if (presence.length != config.classes.size()) {
            throw new IllegalArgumentException("Unexpected presence count");
        }

        Map<Long, float[]> current = new LinkedHashMap<>();
        for (int pairIndex = 0; pairIndex < config.possiblePairs.size(); pairIndex++) {
            ModelConfig.Pair pair = config.possiblePairs.get(pairIndex);
            if (!presence[config.classSlot(pair.subjectId)]
                    || !presence[config.classSlot(pair.objectId)]) {
                continue;
            }
            float[] probabilities = new float[config.verbs.size()];
            for (int verbIndex = 0; verbIndex < probabilities.length; verbIndex++) {
                probabilities[verbIndex] = sigmoid(logits[pairIndex][verbIndex]);
            }
            current.put(TemporalSmoother.pairKey(pair.subjectId, pair.objectId), probabilities);
        }
        return current;
    }

    private List<SemanticRelation> buildRelations(Map<Long, float[]> smoothed) {
        List<SemanticRelation> relations = new ArrayList<>();
        for (Map.Entry<Long, float[]> entry : smoothed.entrySet()) {
            int subjectId = TemporalSmoother.subjectFromKey(entry.getKey());
            int objectId = TemporalSmoother.objectFromKey(entry.getKey());
            float[] probabilities = entry.getValue();
            for (int verbIndex = 0; verbIndex < probabilities.length; verbIndex++) {
                float score = probabilities[verbIndex];
                if (score > config.relationThreshold) {
                    relations.add(new SemanticRelation(
                            subjectId,
                            verbIndex,
                            objectId,
                            config.className(subjectId),
                            config.verbs.get(verbIndex),
                            config.className(objectId),
                            score
                    ));
                }
            }
        }
        relations.sort(Comparator.comparingDouble((SemanticRelation relation) -> relation.score).reversed());
        return relations;
    }

    public void reset() {
        smoother.reset();
    }

    public static float sigmoid(float value) {
        if (value >= 0) {
            double exponential = Math.exp(-value);
            return (float) (1.0 / (1.0 + exponential));
        }
        double exponential = Math.exp(value);
        return (float) (exponential / (1.0 + exponential));
    }
}
