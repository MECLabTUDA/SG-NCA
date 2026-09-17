package com.example.sgnca.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.example.sgnca.R;
import com.example.sgnca.inference.ModelConfig;
import com.example.sgnca.inference.SemanticRelation;
import com.example.sgnca.inference.TouchingRelation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Lightweight frame-local scene graph combining semantic and spatial relations. */
public final class SceneGraphView extends View {
    private final Paint edgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint semanticEdgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint semanticArrowPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint semanticLabelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint semanticLabelBackgroundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint nodePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint nodeOutlinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint emptyPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private List<Node> nodes = Collections.emptyList();
    private List<TouchingRelation> touchingRelations = Collections.emptyList();
    private List<SemanticRelation> semanticRelations = Collections.emptyList();

    public SceneGraphView(Context context) {
        this(context, null);
    }

    public SceneGraphView(Context context, @Nullable AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public SceneGraphView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        edgePaint.setColor(ContextCompat.getColor(context, R.color.mec_on_dark_muted));
        edgePaint.setAlpha(150);
        edgePaint.setStrokeCap(Paint.Cap.ROUND);

        int semanticColor = ContextCompat.getColor(context, R.color.mec_red);
        semanticEdgePaint.setColor(semanticColor);
        semanticEdgePaint.setStrokeCap(Paint.Cap.ROUND);
        semanticEdgePaint.setStyle(Paint.Style.STROKE);

        semanticArrowPaint.setColor(semanticColor);
        semanticArrowPaint.setStyle(Paint.Style.FILL);

        semanticLabelBackgroundPaint.setColor(semanticColor);
        semanticLabelBackgroundPaint.setStyle(Paint.Style.FILL);

        semanticLabelPaint.setColor(ContextCompat.getColor(context, R.color.white));
        semanticLabelPaint.setTextAlign(Paint.Align.CENTER);
        semanticLabelPaint.setTextSize(sp(10));
        semanticLabelPaint.setFakeBoldText(true);

        nodeOutlinePaint.setStyle(Paint.Style.STROKE);
        nodeOutlinePaint.setStrokeWidth(dp(2));
        nodeOutlinePaint.setColor(ContextCompat.getColor(context, R.color.white));

        labelPaint.setColor(ContextCompat.getColor(context, R.color.mec_on_dark));
        labelPaint.setTextAlign(Paint.Align.CENTER);
        labelPaint.setTextSize(sp(11));
        labelPaint.setFakeBoldText(true);

        emptyPaint.setColor(ContextCompat.getColor(context, R.color.mec_on_dark_muted));
        emptyPaint.setTextAlign(Paint.Align.CENTER);
        emptyPaint.setTextSize(sp(14));
        setContentDescription(context.getString(R.string.scene_graph_empty));
    }

    public void setGraph(
            ModelConfig config,
            List<String> detectedObjects,
            List<TouchingRelation> touching,
            List<SemanticRelation> semantic
    ) {
        Set<String> detected = new HashSet<>(detectedObjects);
        Set<Integer> semanticObjectIds = new HashSet<>();
        for (SemanticRelation relation : semantic) {
            semanticObjectIds.add(relation.subjectId);
            semanticObjectIds.add(relation.objectId);
        }
        List<Node> visibleNodes = new ArrayList<>();
        for (ModelConfig.ClassInfo classInfo : config.classes) {
            if (detected.contains(classInfo.name) || semanticObjectIds.contains(classInfo.id)) {
                visibleNodes.add(new Node(classInfo.id, classInfo.name, classInfo.color));
            }
        }
        nodes = Collections.unmodifiableList(visibleNodes);
        touchingRelations = Collections.unmodifiableList(new ArrayList<>(touching));
        semanticRelations = Collections.unmodifiableList(new ArrayList<>(semantic));
        setContentDescription(getResources().getQuantityString(
                R.plurals.scene_graph_description,
                nodes.size(),
                nodes.size(),
                semanticRelations.size(),
                touchingRelations.size()
        ));
        invalidate();
    }

    public void clear() {
        nodes = Collections.emptyList();
        touchingRelations = Collections.emptyList();
        semanticRelations = Collections.emptyList();
        setContentDescription(getContext().getString(R.string.scene_graph_empty));
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (nodes.isEmpty()) {
            canvas.drawText(
                    getContext().getString(R.string.scene_graph_empty),
                    getWidth() / 2.0f,
                    getHeight() / 2.0f,
                    emptyPaint
            );
            return;
        }

        float centerX = getWidth() / 2.0f;
        float centerY = getHeight() / 2.0f;
        float horizontalRadius = Math.max(dp(36), centerX - dp(50));
        float verticalRadius = Math.max(dp(36), centerY - dp(56));
        Map<Integer, PointF> positions = new HashMap<>();
        if (nodes.size() == 1) {
            positions.put(nodes.get(0).id, new PointF(centerX, centerY));
        }
        for (int index = 0; index < nodes.size(); index++) {
            if (nodes.size() == 1) {
                break;
            }
            double angle = -Math.PI / 2.0 + 2.0 * Math.PI * index / nodes.size();
            positions.put(nodes.get(index).id, new PointF(
                    centerX + horizontalRadius * (float) Math.cos(angle),
                    centerY + verticalRadius * (float) Math.sin(angle)
            ));
        }

        for (TouchingRelation relation : touchingRelations) {
            PointF first = positions.get(relation.firstId);
            PointF second = positions.get(relation.secondId);
            if (first == null || second == null) {
                continue;
            }
            float strength = (float) Math.log10(relation.borderContacts + 1.0);
            edgePaint.setStrokeWidth(dp(1.0f + Math.min(2.5f, strength * 0.65f)));
            canvas.drawLine(first.x, first.y, second.x, second.y, edgePaint);
        }

        float nodeRadius = dp(14);
        Map<Long, List<SemanticRelation>> semanticGroups = groupSemanticRelations();
        for (List<SemanticRelation> group : semanticGroups.values()) {
            drawSemanticEdge(canvas, positions, group, nodeRadius);
        }

        for (Node node : nodes) {
            PointF position = positions.get(node.id);
            nodePaint.setColor(node.color);
            nodePaint.setStyle(Paint.Style.FILL);
            canvas.drawCircle(position.x, position.y, nodeRadius, nodePaint);
            canvas.drawCircle(position.x, position.y, nodeRadius, nodeOutlinePaint);
            drawNodeLabel(canvas, node.name, position, centerY, nodeRadius);
        }
    }

    private Map<Long, List<SemanticRelation>> groupSemanticRelations() {
        Map<Long, List<SemanticRelation>> groups = new LinkedHashMap<>();
        for (SemanticRelation relation : semanticRelations) {
            long key = ((long) relation.subjectId << 32) | (relation.objectId & 0xffffffffL);
            List<SemanticRelation> group = groups.get(key);
            if (group == null) {
                group = new ArrayList<>();
                groups.put(key, group);
            }
            group.add(relation);
        }
        return groups;
    }

    private void drawSemanticEdge(
            Canvas canvas,
            Map<Integer, PointF> positions,
            List<SemanticRelation> relations,
            float nodeRadius
    ) {
        if (relations.isEmpty()) {
            return;
        }
        SemanticRelation strongest = relations.get(0);
        PointF subject = positions.get(strongest.subjectId);
        PointF object = positions.get(strongest.objectId);
        if (subject == null || object == null) {
            return;
        }

        float dx = object.x - subject.x;
        float dy = object.y - subject.y;
        float distance = (float) Math.hypot(dx, dy);
        if (distance < nodeRadius * 2.0f) {
            return;
        }
        float unitX = dx / distance;
        float unitY = dy / distance;
        float normalX = -unitY;
        float normalY = unitX;
        float lineOffset = dp(6);
        float endpointInset = nodeRadius + dp(3);
        float startX = subject.x + unitX * endpointInset + normalX * lineOffset;
        float startY = subject.y + unitY * endpointInset + normalY * lineOffset;
        float endX = object.x - unitX * endpointInset + normalX * lineOffset;
        float endY = object.y - unitY * endpointInset + normalY * lineOffset;

        semanticEdgePaint.setStrokeWidth(dp(2.3f + Math.min(1.7f, strongest.score * 1.7f)));
        canvas.drawLine(startX, startY, endX, endY, semanticEdgePaint);
        drawArrowHead(canvas, endX, endY, unitX, unitY);

        float labelX = (startX + endX) / 2.0f + normalX * dp(9);
        float labelY = (startY + endY) / 2.0f + normalY * dp(9);
        drawSemanticLabel(canvas, relations, labelX, labelY);
    }

    private void drawArrowHead(
            Canvas canvas,
            float tipX,
            float tipY,
            float unitX,
            float unitY
    ) {
        float arrowLength = dp(9);
        float arrowHalfWidth = dp(5);
        float baseX = tipX - unitX * arrowLength;
        float baseY = tipY - unitY * arrowLength;
        float normalX = -unitY;
        float normalY = unitX;
        Path arrow = new Path();
        arrow.moveTo(tipX, tipY);
        arrow.lineTo(baseX + normalX * arrowHalfWidth, baseY + normalY * arrowHalfWidth);
        arrow.lineTo(baseX - normalX * arrowHalfWidth, baseY - normalY * arrowHalfWidth);
        arrow.close();
        canvas.drawPath(arrow, semanticArrowPaint);
    }

    private void drawSemanticLabel(
            Canvas canvas,
            List<SemanticRelation> relations,
            float desiredCenterX,
            float desiredCenterY
    ) {
        List<String> lines = new ArrayList<>();
        float widestLine = 0.0f;
        for (SemanticRelation relation : relations) {
            String line = String.format(
                    Locale.US,
                    "%s %.0f%%",
                    relation.verb.replace('_', ' '),
                    relation.score * 100.0f
            );
            lines.add(line);
            widestLine = Math.max(widestLine, semanticLabelPaint.measureText(line));
        }

        Paint.FontMetrics metrics = semanticLabelPaint.getFontMetrics();
        float lineHeight = metrics.descent - metrics.ascent;
        float horizontalPadding = dp(6);
        float verticalPadding = dp(4);
        float labelWidth = widestLine + horizontalPadding * 2.0f;
        float labelHeight = lineHeight * lines.size() + verticalPadding * 2.0f;
        float centerX = Math.max(
                getPaddingLeft() + labelWidth / 2.0f,
                Math.min(getWidth() - getPaddingRight() - labelWidth / 2.0f, desiredCenterX)
        );
        float centerY = Math.max(
                getPaddingTop() + labelHeight / 2.0f,
                Math.min(getHeight() - getPaddingBottom() - labelHeight / 2.0f, desiredCenterY)
        );
        RectF background = new RectF(
                centerX - labelWidth / 2.0f,
                centerY - labelHeight / 2.0f,
                centerX + labelWidth / 2.0f,
                centerY + labelHeight / 2.0f
        );
        canvas.drawRoundRect(background, dp(6), dp(6), semanticLabelBackgroundPaint);

        float baseline = background.top + verticalPadding - metrics.ascent;
        for (String line : lines) {
            canvas.drawText(line, centerX, baseline, semanticLabelPaint);
            baseline += lineHeight;
        }
    }

    private void drawNodeLabel(
            Canvas canvas,
            String rawName,
            PointF position,
            float centerY,
            float nodeRadius
    ) {
        String[] lines = rawName.split("_", 2);
        float lineHeight = labelPaint.getTextSize() * 1.12f;
        boolean drawBelow = position.y <= centerY;
        float firstBaseline = drawBelow
                ? position.y + nodeRadius + lineHeight
                : position.y - nodeRadius - (lines.length - 1) * lineHeight - dp(4);
        for (int index = 0; index < lines.length; index++) {
            float halfWidth = labelPaint.measureText(lines[index]) / 2.0f;
            float textX = Math.max(
                    getPaddingLeft() + halfWidth,
                    Math.min(getWidth() - getPaddingRight() - halfWidth, position.x)
            );
            canvas.drawText(lines[index], textX, firstBaseline + index * lineHeight, labelPaint);
        }
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
    }

    private float sp(float value) {
        return value * getResources().getDisplayMetrics().scaledDensity;
    }

    private static final class Node {
        final int id;
        final String name;
        final int color;

        Node(int id, String name, int color) {
            this.id = id;
            this.name = name;
            this.color = color;
        }
    }
}
