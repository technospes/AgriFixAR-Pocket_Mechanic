import 'dart:ui';
import 'package:flutter/material.dart';

// ── Status enum ──────────────────────────────────────────────────
enum StepStatus { pending, active, completed }

class AnalysisStep {
  final String label;
  StepStatus status;
  AnalysisStep({required this.label, this.status = StepStatus.pending});
}

// ── Public entry point ───────────────────────────────────────────
/// Call this from your "Find Solution" button.
///
/// [runAnalysis] receives a [markDone(stepIndex)] callback.
/// Call markDone(0), markDone(1), markDone(2), markDone(3) as each
/// backend stage completes. Steps can arrive in any order.
///
/// [onCompleted] fires automatically after all 4 steps are done.
///
/// Example:
///   showAnalysisSheet(
///     context,
///     runAnalysis: (markDone) async {
///       await ApiService.diagnose(onStepDone: markDone);
///     },
///     onCompleted: () => context.go('/solution'),
///   );
Future<void> showAnalysisSheet(
  BuildContext context, {
  required Future<void> Function(
    void Function(int stepIndex) markActive,
    void Function(int stepIndex) markDone,
  ) runAnalysis,
  required VoidCallback onCompleted,
}) async {
  await showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    // FIX: transparent so BackdropFilter is visible
    barrierColor: Colors.transparent,
    // REFINEMENT 1: prevent accidental swipe-dismiss mid-analysis
    isDismissible: false,
    enableDrag: false,
    builder: (_) => _AnalysisSheetWithOverlay(
      runAnalysis: runAnalysis,
      onCompleted: onCompleted,
    ),
  );
}

// ── Overlay wrapper ──────────────────────────────────────────────
// Covers the full screen with blur + dark scrim behind the panel.
class _AnalysisSheetWithOverlay extends StatelessWidget {
  final Future<void> Function(
    void Function(int) markActive,
    void Function(int) markDone,
  ) runAnalysis;
  final VoidCallback onCompleted;

  const _AnalysisSheetWithOverlay({
    required this.runAnalysis,
    required this.onCompleted,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Full-screen blur + dim layer
        BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
          // FIX: .withValues() instead of deprecated .withOpacity()
          child: Container(color: Colors.black.withValues(alpha: 0.35)),
        ),
        // Sheet panel anchored to bottom
        Align(
          alignment: Alignment.bottomCenter,
          child: _AnalysisSheet(
            runAnalysis: runAnalysis,
            onCompleted: onCompleted,
          ),
        ),
      ],
    );
  }
}

// ── Bottom Sheet ─────────────────────────────────────────────────
class _AnalysisSheet extends StatefulWidget {
  final Future<void> Function(
    void Function(int) markActive,
    void Function(int) markDone,
  ) runAnalysis;
  final VoidCallback onCompleted;

  const _AnalysisSheet({
    required this.runAnalysis,
    required this.onCompleted,
  });

  @override
  State<_AnalysisSheet> createState() => _AnalysisSheetState();
}

class _AnalysisSheetState extends State<_AnalysisSheet>
    with TickerProviderStateMixin {

  // REFINEMENT 4+5: strings in one place for easy l10n migration later
  static const _stepLabels = [
    'Analyzing machinery type...',
    'Transcribing audio complaint...',
    'Querying repair manuals...',
    'Generating step-by-step guide...',
  ];

  late final List<AnalysisStep> _steps;
  late final List<AnimationController> _checkControllers;
  late final List<Animation<double>> _checkScales;
  late final List<AnimationController> _spinControllers;

  double _progress = 0.0;
  bool _hasError = false;

  @override
  void initState() {
    super.initState();

    _steps = _stepLabels.map((l) => AnalysisStep(label: l)).toList();

    _checkControllers = List.generate(
      _steps.length,
      (_) => AnimationController(
        vsync: this,
        duration: const Duration(milliseconds: 280),
      ),
    );

    _checkScales = _checkControllers.map((c) {
      return Tween<double>(begin: 0.6, end: 1.0).animate(
        CurvedAnimation(parent: c, curve: Curves.easeOutBack),
      );
    }).toList();

    // FIX 1: No ..repeat() here — only spin when step is active
    _spinControllers = List.generate(
      _steps.length,
      (_) => AnimationController(
        vsync: this,
        duration: const Duration(milliseconds: 1400),
      ),
    );

    WidgetsBinding.instance.addPostFrameCallback((_) {
      // backend fires stage_start(0) immediately → markActive(0) drives the spinner
      widget.runAnalysis(_setActive, _markDone).catchError((error) {
        if (!mounted) return;
        // REFINEMENT 3: stop all spinning if error occurs
        for (final c in _spinControllers) {
          c.stop();
        }
        setState(() => _hasError = true);
      });
    });
  }

  @override
  void dispose() {
    for (final c in _checkControllers) c.dispose();
    for (final c in _spinControllers) c.dispose();
    super.dispose();
  }

  // FIX 1: Only the newly active step starts its spinner
  void _setActive(int i) {
    if (!mounted || i >= _steps.length) return;
    setState(() => _steps[i].status = StepStatus.active);
    _spinControllers[i].repeat(); // start spin only for this step
  }

  void _markDone(int i) {
    if (!mounted || i >= _steps.length) return;
    // REFINEMENT 2: guard against double markDone calls
    if (_steps[i].status == StepStatus.completed) return;

    setState(() {
      _steps[i].status = StepStatus.completed;
      _spinControllers[i].stop(); // stop spin for this step
      _progress =
          _steps.where((s) => s.status == StepStatus.completed).length /
              _steps.length;
    });
    _checkControllers[i].forward();

    final next = i + 1;
    if (next < _steps.length) {
      Future.delayed(const Duration(milliseconds: 150), () => _setActive(next));
    } else {
      // All steps complete → brief pause → close + navigate
      Future.delayed(const Duration(milliseconds: 600), () {
        if (mounted) Navigator.of(context).pop();
        widget.onCompleted();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        boxShadow: [
          BoxShadow(
            color: Colors.black26,
            blurRadius: 28,
            offset: Offset(0, -6),
          ),
        ],
      ),
      padding: const EdgeInsets.fromLTRB(24, 12, 24, 40),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // ── Drag handle ──
          Container(
            width: 36,
            height: 4,
            decoration: BoxDecoration(
              color: const Color(0xFFD1D5DB),
              borderRadius: BorderRadius.circular(8),
            ),
          ),
          const SizedBox(height: 24),

          // ── Title ──
          const Text(
            'AI Analysis in Progress',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: Color(0xFF1F2937),
            ),
          ),
          const SizedBox(height: 8),

          // ── Subtitle ──
          const Text(
            'Please wait while we analyze your machine issue.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w400,
              color: Color(0xFF6B7280),
            ),
          ),
          const SizedBox(height: 28),

          // ── Error state (FIX 5) ──
          if (_hasError) ...[
            const Icon(
              Icons.error_outline_rounded,
              color: Color(0xFFEF4444),
              size: 40,
            ),
            const SizedBox(height: 12),
            const Text(
              'Analysis failed. Please try again.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: Color(0xFFEF4444),
              ),
            ),
            const SizedBox(height: 20),
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Dismiss'),
            ),
          ] else ...[

            // ── Checklist ──
            ...List.generate(
              _steps.length,
              (i) => _StepRow(
                step: _steps[i],
                spinController: _spinControllers[i],
                checkScale: _checkScales[i],
              ),
            ),
            const SizedBox(height: 24),

            // ── Progress bar ──
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: TweenAnimationBuilder<double>(
                tween: Tween(begin: 0, end: _progress),
                duration: const Duration(milliseconds: 450),
                curve: Curves.easeInOut,
                builder: (_, val, __) => LinearProgressIndicator(
                  value: val,
                  minHeight: 6,
                  backgroundColor: const Color(0xFFE5E7EB),
                  valueColor:
                      const AlwaysStoppedAnimation(Color(0xFF10B981)),
                ),
              ),
            ),
            const SizedBox(height: 8),

            // ── Progress % label ──
            Align(
              alignment: Alignment.centerRight,
              child: Text(
                '${(_progress * 100).toInt()}%',
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFF10B981),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// ── Single Step Row ──────────────────────────────────────────────
class _StepRow extends StatelessWidget {
  final AnalysisStep step;
  final AnimationController spinController;
  final Animation<double> checkScale;

  const _StepRow({
    required this.step,
    required this.spinController,
    required this.checkScale,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: Row(
        children: [
          SizedBox(width: 32, height: 32, child: _icon()),
          const SizedBox(width: 14),
          Expanded(
            child: Text(
              step.label,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: step.status == StepStatus.pending
                    ? const Color(0xFF9CA3AF)
                    : const Color(0xFF1F2937),
              ),
            ),
          ),
          _tag(),
        ],
      ),
    );
  }

  Widget _icon() {
    switch (step.status) {
      case StepStatus.pending:
        return Container(
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: const Color(0xFFD1D5DB), width: 2),
          ),
        );

      case StepStatus.active:
        // REFINEMENT 4: CurvedAnimation with linear curve for smoother spin
        return RotationTransition(
          turns: CurvedAnimation(
            parent: spinController,
            curve: Curves.linear,
          ),
          child: const Icon(
            Icons.timelapse_rounded,
            color: Color(0xFF10B981),
            size: 24, // FIX 3: 24px fits cleanly in 32px container
          ),
        );

      case StepStatus.completed:
        return ScaleTransition(
          scale: checkScale,
          child: Container(
            decoration: const BoxDecoration(
              shape: BoxShape.circle,
              color: Color(0xFF10B981),
            ),
            child: const Icon(
              Icons.check_rounded,
              color: Colors.white,
              size: 20,
            ),
          ),
        );
    }
  }

  Widget _tag() {
    switch (step.status) {
      case StepStatus.pending:
        return const SizedBox(width: 76);

      case StepStatus.active:
        return SizedBox(
          width: 76,
          child: Text(
            'PROCESSING',
            textAlign: TextAlign.right,
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.3,
              color: Colors.orange.shade700,
            ),
          ),
        );

      case StepStatus.completed:
        return const SizedBox(
          width: 76,
          child: Text(
            'DONE',
            textAlign: TextAlign.right,
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.3,
              color: Color(0xFF10B981),
            ),
          ),
        );
    }
  }
}