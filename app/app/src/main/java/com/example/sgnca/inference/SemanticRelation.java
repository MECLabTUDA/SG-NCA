package com.example.sgnca.inference;

public final class SemanticRelation {
    public final int subjectId;
    public final int verbIndex;
    public final int objectId;
    public final String subject;
    public final String verb;
    public final String object;
    public final float score;

    SemanticRelation(
            int subjectId,
            int verbIndex,
            int objectId,
            String subject,
            String verb,
            String object,
            float score
    ) {
        this.subjectId = subjectId;
        this.verbIndex = verbIndex;
        this.objectId = objectId;
        this.subject = subject;
        this.verb = verb;
        this.object = object;
        this.score = score;
    }

    public String sentence() {
        return subject + " " + verb + "s " + object;
    }
}
