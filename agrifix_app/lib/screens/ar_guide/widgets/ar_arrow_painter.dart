// lib/screens/ar_guide/widgets/ar_arrow_painter.dart
// ignore_for_file: deprecated_member_use
//
// AR Arrow CustomPainter.
// Renders over the full camera preview (not clipped to scan box):
//   1. Pulsing bbox rectangle + white shadow for visibility on any surface
//   2. Corner bracket accents (28px, white shadow behind green)
//   3. Dashed arrow from screen bottom-centre to part edge
//   4. Part-label chip on arrow shaft
//
// All rendering is in screen pixels. NormBbox (0–1) is converted here.
// No native AR/Unity SDK — pure Flutter CustomPainter.

import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../models/bbox.dart';

class ARArrowPainter extends CustomPainter {
  final NormBbox bbox;
  final double   previewW, previewH;
  final double   pulseValue;   // 0.0–1.0 from repeat(reverse:true) controller
  final String   partLabel;
  final bool     isHindi;

  static const _green = Color(0xFF22C55E);

  const ARArrowPainter({
    required this.bbox,
    required this.previewW,
    required this.previewH,
    required this.pulseValue,
    required this.partLabel,
    required this.isHindi,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final partRect = bbox.toScreenRect(previewW, previewH);

    // ── 1. Bbox fill + border ─────────────────────────────────────────────
    final alpha = 0.55 + 0.30 * pulseValue;
    final fillPaint = Paint()
      ..color = _green.withOpacity(0.18 + 0.10 * pulseValue)
      ..style = PaintingStyle.fill;
    final borderPaint = Paint()
      ..color       = _green.withOpacity(alpha)
      ..style       = PaintingStyle.stroke
      ..strokeWidth = 4.0;

    final rRect = RRect.fromRectAndRadius(partRect, const Radius.circular(10));
    canvas.drawRRect(rRect, fillPaint);
    // White shadow — ensures visibility on rusty red, dark metal, any surface
    canvas.drawRRect(rRect, Paint()
      ..color       = Colors.white.withOpacity(0.90)
      ..style       = PaintingStyle.stroke
      ..strokeWidth = 6.5);
    canvas.drawRRect(rRect, borderPaint);

    // ── 2. Corner bracket accents ─────────────────────────────────────────
    _drawCornerBrackets(canvas, partRect, _green.withOpacity(0.85 + 0.15 * pulseValue));

    // ── 3. Dashed arrow ───────────────────────────────────────────────────
    final tail = Offset(size.width / 2, size.height * 0.72);
    final head = _nearestEdgePoint(partRect, tail);

    final arrowShadow = Paint()
      ..color       = Colors.white.withOpacity(0.85)
      ..strokeWidth = 6.0
      ..strokeCap   = StrokeCap.round
      ..style       = PaintingStyle.stroke;
    final arrowPaint = Paint()
      ..color       = _green.withOpacity(0.95)
      ..strokeWidth = 3.5
      ..strokeCap   = StrokeCap.round
      ..style       = PaintingStyle.stroke;

    _drawDashedLine(canvas, tail, head, arrowShadow);
    _drawDashedLine(canvas, tail, head, arrowPaint);
    _drawArrowhead(canvas, tail, head);

    // ── 4. Part-label chip on shaft ───────────────────────────────────────
    if (partLabel.isNotEmpty) {
      _drawChip(canvas, size, Offset(
        (tail.dx + head.dx) / 2,
        (tail.dy + head.dy) / 2,
      ), partLabel);
    }
  }

  void _drawCornerBrackets(Canvas canvas, Rect rect, Color color) {
    final shadow = Paint()
      ..color       = Colors.white.withOpacity(0.90)
      ..strokeWidth = 6.0
      ..strokeCap   = StrokeCap.round
      ..style       = PaintingStyle.stroke;
    final p = Paint()
      ..color       = color
      ..strokeWidth = 4.0
      ..strokeCap   = StrokeCap.round
      ..style       = PaintingStyle.stroke;
    const L = 28.0;
    final corners = [
      [Offset(rect.left, rect.top + L),      Offset(rect.left,  rect.top),      Offset(rect.left  + L, rect.top)],
      [Offset(rect.right - L, rect.top),     Offset(rect.right, rect.top),      Offset(rect.right,     rect.top + L)],
      [Offset(rect.left, rect.bottom - L),   Offset(rect.left,  rect.bottom),   Offset(rect.left  + L, rect.bottom)],
      [Offset(rect.right - L, rect.bottom),  Offset(rect.right, rect.bottom),   Offset(rect.right,     rect.bottom - L)],
    ];
    for (final paint in [shadow, p]) {
      for (final c in corners) {
        canvas.drawPath(
          Path()
            ..moveTo(c[0].dx, c[0].dy)
            ..lineTo(c[1].dx, c[1].dy)
            ..lineTo(c[2].dx, c[2].dy),
          paint,
        );
      }
    }
  }

  void _drawDashedLine(Canvas canvas, Offset from, Offset to, Paint paint) {
    const dashLen = 10.0, gapLen = 6.0;
    final dx = to.dx - from.dx, dy = to.dy - from.dy;
    final total = math.sqrt(dx * dx + dy * dy);
    if (total < 1) return;
    final nx = dx / total, ny = dy / total;
    double traveled = 0;
    bool drawing = true;
    while (traveled < total) {
      final seg = drawing ? dashLen : gapLen;
      final end = math.min(traveled + seg, total);
      if (drawing) {
        canvas.drawLine(
          Offset(from.dx + nx * traveled, from.dy + ny * traveled),
          Offset(from.dx + nx * end,      from.dy + ny * end),
          paint,
        );
      }
      traveled += seg;
      drawing = !drawing;
    }
  }

  void _drawArrowhead(Canvas canvas, Offset tail, Offset head) {
    final dx = head.dx - tail.dx, dy = head.dy - tail.dy;
    final len = math.sqrt(dx * dx + dy * dy);
    if (len < 1) return;
    final nx = dx / len, ny = dy / len;
    const sz = 16.0, hw = 9.0;
    final base  = Offset(head.dx - nx * sz, head.dy - ny * sz);
    final left  = Offset(base.dx + ny * hw, base.dy - nx * hw);
    final right = Offset(base.dx - ny * hw, base.dy + nx * hw);
    canvas.drawPath(
      Path()
        ..moveTo(head.dx, head.dy)
        ..lineTo(left.dx, left.dy)
        ..lineTo(right.dx, right.dy)
        ..close(),
      Paint()..color = _green..style = PaintingStyle.fill,
    );
  }

  void _drawChip(Canvas canvas, Size canvasSize, Offset centre, String label) {
    final display = label.length > 28 ? '${label.substring(0, 25)}…' : label;
    final tp = TextPainter(
      text: TextSpan(
        text: display,
        style: const TextStyle(
          color: Colors.white, fontSize: 12,
          fontWeight: FontWeight.w600, letterSpacing: 0.4,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    const pH = 12.0, pV = 7.0;
    final cW = tp.width + pH * 2, cH = tp.height + pV * 2;
    final cx = centre.dx.clamp(cW / 2 + 8, canvasSize.width  - cW / 2 - 8);
    final cy = centre.dy.clamp(cH / 2 + 8, canvasSize.height - cH / 2 - 8);
    final r  = RRect.fromRectAndRadius(
        Rect.fromCenter(center: Offset(cx, cy), width: cW, height: cH),
        const Radius.circular(20));

    canvas.drawRRect(r, Paint()..color = Colors.black.withOpacity(0.72));
    canvas.drawRRect(r, Paint()
      ..color       = _green.withOpacity(0.60)
      ..style       = PaintingStyle.stroke
      ..strokeWidth = 1.2);
    tp.paint(canvas, Offset(cx - tp.width / 2, cy - tp.height / 2));
  }

  Offset _nearestEdgePoint(Rect rect, Offset from) {
    final cx = rect.left + rect.width  / 2;
    final cy = rect.top  + rect.height / 2;
    final dx = from.dx - cx, dy = from.dy - cy;
    final len = math.sqrt(dx * dx + dy * dy);
    if (len < 1) return rect.center;
    final nx = dx / len, ny = dy / len;
    final tx = nx != 0 ? (nx > 0 ? rect.right  - cx : rect.left - cx) / nx : double.infinity;
    final ty = ny != 0 ? (ny > 0 ? rect.bottom - cy : rect.top  - cy) / ny : double.infinity;
    final t  = math.min(tx.abs(), ty.abs());
    return Offset(cx + nx * t, cy + ny * t);
  }

  @override
  bool shouldRepaint(ARArrowPainter old) =>
      old.bbox       != bbox       ||
      old.pulseValue != pulseValue ||
      old.partLabel  != partLabel;
}
