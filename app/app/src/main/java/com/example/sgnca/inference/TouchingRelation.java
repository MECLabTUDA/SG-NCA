package com.example.sgnca.inference;

/** An undirected, frame-local relation between two visible segmentation classes. */
public final class TouchingRelation {
    public final int firstId;
    public final int secondId;
    public final String first;
    public final String second;
    public final int borderContacts;

    TouchingRelation(
            int firstId,
            int secondId,
            String first,
            String second,
            int borderContacts
    ) {
        this.firstId = firstId;
        this.secondId = secondId;
        this.first = first;
        this.second = second;
        this.borderContacts = borderContacts;
    }
}
