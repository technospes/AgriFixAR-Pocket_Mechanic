// ignore_for_file: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'package:go_router/go_router.dart';                  // AR nav
import '../../core/providers/diagnosis_provider.dart';
import '../../core/router.dart';                      // AppRoutes

// ═══════════════════════════════════════════════════════════════════════════
// SolutionScreen  —  "AI Repair Guide"  (Text Solution Scene)
//
// NEW in this version:
//   • Prev/Next row pinned ABOVE the AR button (not inside scroll)
//   • Voice button: tap → speaks current step body (TTS)
//                   tap again → stops speech
//                   tap again → speaks current step again
//   • Voice icon changes: volume_up (unmuted/speaking) ↔ volume_off (muted)
//   • Auto-speaks when step changes while already speaking
// ═══════════════════════════════════════════════════════════════════════════
class SolutionScreen extends StatefulWidget {
  const SolutionScreen({super.key});

  @override
  State<SolutionScreen> createState() => _SolutionScreenState();
}

class _SolutionScreenState extends State<SolutionScreen>
    with TickerProviderStateMixin {
  int _currentStep = 0;
  final ScrollController _scroll = ScrollController();
  final List<GlobalKey> _stepKeys = [];

  // Entrance animation replays every time active step changes
  late final AnimationController _entranceCtrl;

  // ── TTS ──────────────────────────────────────────────────────────────
  final FlutterTts _tts = FlutterTts();
  bool _isSpeaking = false;   // true while TTS engine is playing

  @override
  void initState() {
    super.initState();
    _entranceCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 320),
    )..forward();

    _initTts();
  }

  Future<void> _initTts() async {
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.48);   // slightly slower — easier for farmers
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);

    // Keep _isSpeaking in sync with actual TTS state
    _tts.setStartHandler(() {
      if (mounted) setState(() => _isSpeaking = true);
    });
    _tts.setCompletionHandler(() {
      if (mounted) setState(() => _isSpeaking = false);
    });
    _tts.setCancelHandler(() {
      if (mounted) setState(() => _isSpeaking = false);
    });
    _tts.setErrorHandler((_) {
      if (mounted) setState(() => _isSpeaking = false);
    });
  }

  /// Speak a string, stopping any current speech first.
  Future<void> _speak(String text) async {
    await _tts.stop();
    await _tts.speak(text);
  }

  /// Toggle: if currently speaking → stop; if muted → speak current step.
  Future<void> _toggleVoice(List<StepData> steps) async {
    HapticFeedback.lightImpact();
    
    if (_isSpeaking) {
      // 🔥 OPTIMISTIC UPDATE: Force UI to false instantly
      setState(() => _isSpeaking = false); 
      await _tts.stop();
    } else {
      if (_currentStep < steps.length) {
        // 🔥 OPTIMISTIC UPDATE: Force UI to true instantly
        setState(() => _isSpeaking = true); 
        final text = _buildSpeakText(steps[_currentStep], _currentStep + 1);
        await _tts.stop(); // Clear any stuck queues
        await _tts.speak(text);
      }
    }
  }
  /// Full spoken text for a step: "Step 1. [title]. [body]"
  String _buildSpeakText(StepData step, int stepNum) {
    final title = _stepTitle(step);
    final body  = _stepBody(step);
    return 'Step $stepNum. $title. $body';
  }

  @override
  void dispose() {
    _tts.stop();
    _scroll.dispose();
    _entranceCtrl.dispose();
    super.dispose();
  }

  // ── Data helpers ──────────────────────────────────────────────────────
  List<StepData> _stepsFromProvider(DiagnosisProvider prov) =>
      prov.solution?.steps ?? [];

  String _problemTitleFromProvider(DiagnosisProvider prov) =>
      prov.problemDescription ??
      prov.solution?.problemIdentified ??
      'Machine issue detected';

  String _stepBody(StepData step) =>
      step.textEn.isNotEmpty ? step.textEn : step.text;

  String _stepTitle(StepData step) {
    const partTitles = {
      'ignition_key':       'Check the ignition key',
      'battery_terminal':   'Inspect battery terminals',
      'fuel_cap':           'Check the fuel cap',
      'fan_belt':           'Examine the fan belt',
      'clutch_pedal':       'Inspect the clutch pedal',
      'clutch_cable':       'Check the clutch cable',
      'gear_lever':         'Check the gear lever',
      'air_filter':         'Examine the air filter',
      'spark_plug':         'Check the spark plug',
      'radiator_cap':       'Inspect the radiator cap',
      'engine_oil_dipstick':'Check engine oil level',
      'wiring_harness':     'Inspect the wiring',
      'hydraulic_pump':     'Check the hydraulic pump',
      'fuel_filter':        'Locate the main Fuel Filter assembly',
      'drain_plug':         'Unscrew the top drain plug carefully',
    };
    final part = step.visualCue ?? '';
    if (partTitles.containsKey(part)) return partTitles[part]!;
    final body = _stepBody(step);
    if (body.isEmpty) return 'Perform inspection';
    final cut = body.lastIndexOf(' ', 44);
    return cut > 10 ? '${body.substring(0, cut)}…'
                    : '${body.substring(0, body.length.clamp(0, 44))}…';
  }

  String _stepEmoji(StepData step) {
    const m = {
      'ignition_key': '🔑', 'battery_terminal': '🔋', 'fuel_cap': '⛽',
      'fan_belt': '⚙️',     'clutch_pedal': '🦶',      'clutch_cable': '🔗',
      'gear_lever': '🎚️',  'air_filter': '💨',          'spark_plug': '⚡',
      'radiator_cap': '🌡️','engine_oil_dipstick': '🛢️','wiring_harness': '🔌',
      'hydraulic_pump': '💧','fuel_filter': '🔧',        'drain_plug': '📷',
    };
    return m[step.visualCue ?? ''] ?? '🔧';
  }

  // ── Navigation ────────────────────────────────────────────────────────
  void _jumpTo(int index, List<StepData> steps) {
    if (index < 0 || index >= steps.length) return;
    HapticFeedback.lightImpact();
    setState(() => _currentStep = index);
    _entranceCtrl..reset()..forward();

    // If voice was active, auto-read the new step
    if (_isSpeaking) {
      _speak(_buildSpeakText(steps[index], index + 1));
    }

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (index < _stepKeys.length) {
        final ctx = _stepKeys[index].currentContext;
        if (ctx != null) {
          Scrollable.ensureVisible(ctx,
              duration: const Duration(milliseconds: 420),
              curve: Curves.easeOutCubic,
              alignment: 0.08);
        }
      }
    });
  }

  // ── Build ─────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final prov    = context.watch<DiagnosisProvider>();
    final steps   = _stepsFromProvider(prov);
    final problem = _problemTitleFromProvider(prov);

    while (_stepKeys.length < steps.length) _stepKeys.add(GlobalKey());

    final bottomPad = MediaQuery.of(context).padding.bottom;
    // Height of the fixed bottom bar: PrevNext(56) + gap(12) + AR(64) + gap(20) + safeArea
    final fixedBarH = 56 + 12 + 64 + 20 + bottomPad;

    return Scaffold(
      backgroundColor: _SC.bg,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Scrollable content ─────────────────────────────────────
          SafeArea(
            bottom: false,
            child: SingleChildScrollView(
              controller: _scroll,
              padding: EdgeInsets.only(
                left: _SS.outer, right: _SS.outer,
                bottom: fixedBarH + 8,
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 8),

                  // Header  (voice + share live here)
                  _HeaderRow(
                    isSpeaking: _isSpeaking,
                    onBack:  () => Navigator.of(context).pop(),
                    onVoice: () => _toggleVoice(steps),
                  ).animate().fadeIn(duration: 280.ms),

                  const SizedBox(height: 20),

                  _ProgressBars(
                    total: steps.isEmpty ? 4 : steps.length,
                    current: _currentStep,
                  ).animate().fadeIn(duration: 340.ms, delay: 40.ms),

                  const SizedBox(height: 20),

                  _SolutionCard(problem: problem)
                      .animate()
                      .fadeIn(duration: 380.ms, delay: 80.ms)
                      .slideY(begin: 0.06, end: 0,
                              duration: 380.ms, delay: 80.ms,
                              curve: Curves.easeOutCubic),

                  const SizedBox(height: 20),

                  if (steps.isEmpty)
                    _EmptyState()
                  else
                    ...List.generate(steps.length, (i) {
                      return Padding(
                        key: _stepKeys[i],
                        padding: const EdgeInsets.only(bottom: 16),
                        child: _StepCard(
                          stepNumber:  i + 1,
                          title:       _stepTitle(steps[i]),
                          emoji:       _stepEmoji(steps[i]),
                          body:        _stepBody(steps[i]),
                          isActive:    i == _currentStep,
                          isPast:      i < _currentStep,
                          entranceCtrl: i == _currentStep ? _entranceCtrl : null,
                        ),
                      );
                    }),

                  const SizedBox(height: 16),
                ],
              ),
            ),
          ),

          // ── Fixed bottom bar: PrevNext + AR button ─────────────────
          Positioned(
            left: 0, right: 0,
            bottom: 0,
            child: _FixedBottomBar(
              currentStep: _currentStep,
              totalSteps:  steps.length,
              bottomPad:   bottomPad,
              onPrev: () => _jumpTo(_currentStep - 1, steps),
              onNext: () => _jumpTo(_currentStep + 1, steps),
              onAR: () => context.push(AppRoutes.arGuide, extra: _currentStep),
            )
            .animate()
            .slideY(begin: 1.0, end: 0,
                    duration: 480.ms, delay: 200.ms,
                    curve: Curves.easeOutCubic)
            .fadeIn(duration: 380.ms, delay: 200.ms),
          ),
        ],
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _FixedBottomBar  — PrevNext row + AR button, pinned at bottom
// ═══════════════════════════════════════════════════════════════════════════
class _FixedBottomBar extends StatelessWidget {
  final int currentStep;
  final int totalSteps;
  final double bottomPad;
  final VoidCallback onPrev;
  final VoidCallback onNext;
  final VoidCallback onAR;

  const _FixedBottomBar({
    required this.currentStep,
    required this.totalSteps,
    required this.bottomPad,
    required this.onPrev,
    required this.onNext,
    required this.onAR,
  });

  @override
  Widget build(BuildContext context) {
    final canPrev = currentStep > 0;
    final canNext = currentStep < totalSteps - 1;

    return Container(
      // Frosted white strip
      decoration: BoxDecoration(
        color: _SC.bg.withOpacity(0.96),
        border: const Border(
          top: BorderSide(color: Color(0xFFE5E7EB), width: 1),
        ),
      ),
      padding: EdgeInsets.fromLTRB(
        _SS.outer, 14, _SS.outer, 20 + bottomPad),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // ── Prev / Next ────────────────────────────────────────────
          if (totalSteps > 0) ...[
            Row(children: [
              // Prev
              Opacity(
                opacity: canPrev ? 1.0 : 0.28,
                child: GestureDetector(
                  onTap: canPrev ? onPrev : null,
                  behavior: HitTestBehavior.opaque,
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      const Icon(Icons.arrow_back_rounded,
                          size: 18, color: Color(0xFF6B7280)),
                      const SizedBox(width: 5),
                      Text('Prev',
                        style: GoogleFonts.inter(
                          fontSize: 15, fontWeight: FontWeight.w500,
                          color: const Color(0xFF6B7280))),
                    ]),
                  ),
                ),
              ),

              const Spacer(),

              // Step counter in centre  "2 / 5"
              Text(
                totalSteps > 0 ? '${currentStep + 1} / $totalSteps' : '',
                style: GoogleFonts.inter(
                  fontSize: 13, fontWeight: FontWeight.w500,
                  color: const Color(0xFF9CA3AF)),
              ),

              const Spacer(),

              // Next pill
              AnimatedOpacity(
                duration: const Duration(milliseconds: 200),
                opacity: canNext ? 1.0 : 0.28,
                child: GestureDetector(
                  onTap: canNext ? onNext : null,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 22, vertical: 11),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF3F4F6),
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      Text('Next',
                        style: GoogleFonts.inter(
                          fontSize: 15, fontWeight: FontWeight.w600,
                          color: _SC.textDark)),
                      const SizedBox(width: 5),
                      const Icon(Icons.arrow_forward_rounded,
                          size: 18, color: _SC.textDark),
                    ]),
                  ),
                ),
              ),
            ]),
            const SizedBox(height: 12),
          ],

          // ── AR button ──────────────────────────────────────────────
          _ARButton(onTap: onAR),
        ],
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _HeaderRow  —  back · title · voice toggle · share
// ═══════════════════════════════════════════════════════════════════════════
class _HeaderRow extends StatelessWidget {
  final bool isSpeaking;
  final VoidCallback onBack;
  final VoidCallback onVoice;

  const _HeaderRow({
    required this.isSpeaking,
    required this.onBack,
    required this.onVoice,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        GestureDetector(
          onTap: onBack,
          behavior: HitTestBehavior.opaque,
          child: const Padding(
            padding: EdgeInsets.only(right: 8, top: 4, bottom: 4),
            child: Icon(Icons.arrow_back_rounded,
                size: 24, color: _SC.textDark),
          ),
        ),

        Expanded(
          child: Text('AI Repair Guide',
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
              fontSize: 21, fontWeight: FontWeight.w600,
              color: _SC.textDark, letterSpacing: -0.3)),
        ),

        // ── Voice toggle button ──────────────────────────────────────
        GestureDetector(
          onTap: onVoice,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            width: 44, height: 44,
            decoration: BoxDecoration(
              // Speaking → solid green; muted → light green tint
              color: isSpeaking
                  ? _SC.primary
                  : const Color(0xFFE3F5EC),
              shape: BoxShape.circle,
            ),
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 180),
              child: Icon(
                isSpeaking
                    ? Icons.volume_up_rounded    // actively speaking
                    : Icons.volume_off_rounded,  // muted / stopped
                key: ValueKey(isSpeaking),
                size: 22,
                color: isSpeaking ? Colors.white : _SC.primary,
              ),
            ),
          ),
        ),

        const SizedBox(width: 10),

        GestureDetector(
          onTap: () {},
          child: const Icon(Icons.share_outlined,
              size: 20, color: Color(0xFF6B7280)),
        ),
      ],
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _ProgressBars
// ═══════════════════════════════════════════════════════════════════════════
class _ProgressBars extends StatelessWidget {
  final int total;
  final int current;
  const _ProgressBars({required this.total, required this.current});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: List.generate(total, (i) {
        final active = i == current;
        final done   = i < current;
        return Expanded(
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 360),
            curve: Curves.easeInOut,
            margin: EdgeInsets.only(right: i < total - 1 ? 6 : 0),
            height: 5,
            decoration: BoxDecoration(
              color: active ? _SC.primary
                  : done   ? const Color(0xFF86D7A8)
                           : const Color(0xFFE2E4E8),
              borderRadius: BorderRadius.circular(10),
            ),
          ),
        );
      }),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _SolutionCard  (warm beige)
// ═══════════════════════════════════════════════════════════════════════════
class _SolutionCard extends StatelessWidget {
  final String problem;
  const _SolutionCard({required this.problem});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(
          horizontal: _SS.cardPad, vertical: 16),
      decoration: BoxDecoration(
        color: const Color(0xFFF3E9DD),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFE6D2B8), width: 1),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Container(
            width: 42, height: 42,
            decoration: const BoxDecoration(
                color: Color(0xFFFF9F3F), shape: BoxShape.circle),
            child: const Icon(Icons.check_rounded,
                color: Colors.white, size: 22),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('SOLUTION FOUND',
                  style: GoogleFonts.inter(
                    fontSize: 11, fontWeight: FontWeight.w600,
                    color: const Color(0xFF8C8C8C), letterSpacing: 1.3)),
                const SizedBox(height: 3),
                Text(problem,
                  style: GoogleFonts.inter(
                    fontSize: 16, fontWeight: FontWeight.w600,
                    color: _SC.textDark, height: 1.3),
                  maxLines: 2, overflow: TextOverflow.ellipsis),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _StepCard
// ═══════════════════════════════════════════════════════════════════════════
class _StepCard extends StatelessWidget {
  final int stepNumber;
  final String title;
  final String emoji;
  final String body;
  final bool isActive;
  final bool isPast;
  final AnimationController? entranceCtrl;

  const _StepCard({
    required this.stepNumber,
    required this.title,
    required this.emoji,
    required this.body,
    required this.isActive,
    required this.isPast,
    this.entranceCtrl,
  });

  @override
  Widget build(BuildContext context) {
    final opacity = isActive ? 1.0 : (isPast ? 0.50 : 0.38);

    Widget card = AnimatedOpacity(
      duration: const Duration(milliseconds: 300),
      opacity: opacity,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOutCubic,
        padding: const EdgeInsets.all(_SS.cardPad),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(24),
          boxShadow: isActive
              ? [BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 24, offset: const Offset(0, 8))]
              : [],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                decoration: BoxDecoration(
                  color: isActive
                      ? const Color(0xFFE6F6EC)
                      : const Color(0xFFF3F4F6),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Text('Step $stepNumber',
                  style: GoogleFonts.inter(
                    fontSize: 13, fontWeight: FontWeight.w600,
                    color: isActive ? _SC.primary : const Color(0xFF9CA3AF))),
              ),
              const Spacer(),
              if (isActive) ...[
                Container(width: 7, height: 7,
                    decoration: const BoxDecoration(
                        color: _SC.primary, shape: BoxShape.circle)),
                const SizedBox(width: 5),
                Text('Current Task',
                  style: GoogleFonts.inter(
                    fontSize: 13, fontWeight: FontWeight.w500,
                    color: const Color(0xFF86D7A8))),
              ],
            ]),

            const SizedBox(height: 14),

            Text('$title $emoji',
              style: GoogleFonts.inter(
                fontSize: 22, fontWeight: FontWeight.w700,
                color: _SC.textDark, height: 1.3)),

            const SizedBox(height: 16),

            Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 14, vertical: 13),
              decoration: BoxDecoration(
                color: const Color(0xFFF2F4F7),
                borderRadius: BorderRadius.circular(18),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: 26, height: 26,
                    margin: const EdgeInsets.only(top: 1),
                    decoration: BoxDecoration(
                      color: isActive
                          ? const Color(0xFFFF9F3F)
                          : const Color(0xFFD1D5DB),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(Icons.info_outline_rounded,
                      size: 15,
                      color: isActive
                          ? Colors.white
                          : const Color(0xFF9CA3AF)),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(body,
                      style: GoogleFonts.inter(
                        fontSize: 15, fontWeight: FontWeight.w400,
                        color: const Color(0xFF6B7280), height: 1.55)),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );

    if (isActive && entranceCtrl != null) {
      card = AnimatedBuilder(
        animation: entranceCtrl!,
        builder: (_, child) {
          final t = CurvedAnimation(
              parent: entranceCtrl!, curve: Curves.easeOutCubic).value;
          return Opacity(
            opacity: t,
            child: Transform.translate(
                offset: Offset(0, 14 * (1 - t)), child: child),
          );
        },
        child: card,
      );
    }
    return card;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _ARButton
// ═══════════════════════════════════════════════════════════════════════════
class _ARButton extends StatefulWidget {
  final VoidCallback onTap;
  const _ARButton({required this.onTap});

  @override
  State<_ARButton> createState() => _ARButtonState();
}

class _ARButtonState extends State<_ARButton> {
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => setState(() => _pressed = true),
      onTapUp:   (_) { setState(() => _pressed = false); widget.onTap(); },
      onTapCancel: () => setState(() => _pressed = false),
      child: AnimatedScale(
        scale: _pressed ? 0.97 : 1.0,
        duration: const Duration(milliseconds: 90),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 140),
          width: double.infinity,
          height: 64,
          decoration: BoxDecoration(
            color: _pressed ? const Color(0xFF18A34A) : _SC.primary,
            borderRadius: BorderRadius.circular(28),
            boxShadow: _pressed ? [] : [
              BoxShadow(
                color: _SC.primary.withOpacity(0.30),
                blurRadius: 24, offset: const Offset(0, 10)),
            ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.camera_alt_rounded,
                  color: Colors.white, size: 22),
              const SizedBox(width: 10),
              Text('Start AR Guide',
                style: GoogleFonts.inter(
                  fontSize: 18, fontWeight: FontWeight.w600,
                  color: Colors.white, letterSpacing: 0.15)),
            ],
          ),
        ),
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _EmptyState
// ═══════════════════════════════════════════════════════════════════════════
class _EmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
          color: Colors.white, borderRadius: BorderRadius.circular(24)),
      child: Column(children: [
        const Icon(Icons.build_circle_outlined,
            size: 48, color: Color(0xFFD1D5DB)),
        const SizedBox(height: 16),
        Text('No repair steps available',
          style: GoogleFonts.inter(
            fontSize: 16, fontWeight: FontWeight.w600,
            color: const Color(0xFF6B7280))),
        const SizedBox(height: 8),
        Text(
          'The diagnosis completed but no guide was generated.\nPlease try uploading again.',
          textAlign: TextAlign.center,
          style: GoogleFonts.inter(
            fontSize: 14, color: const Color(0xFF9CA3AF), height: 1.55)),
      ]),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Design tokens
// ═══════════════════════════════════════════════════════════════════════════
abstract class _SC {
  static const bg      = Color(0xFFF5F6F7);
  static const primary = Color(0xFF22C55E);
  static const textDark = Color(0xFF111827);
}

abstract class _SS {
  static const double outer   = 24;
  static const double cardPad = 20;
}