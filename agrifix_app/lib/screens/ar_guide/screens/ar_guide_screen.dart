// lib/screens/ar_guide/screens/ar_guide_screen.dart
// ignore_for_file: deprecated_member_use
//
// AR Guide Screen — v4.3 (refactored architecture)
//
// This file is the COMPOSITION layer only:
//   • Owns all AnimationControllers (requires TickerProviderStateMixin)
//   • Creates ARController and passes lifecycle references
//   • Renders the widget tree by connecting controller state to widgets
//   • Handles didChangeDependencies for TTS re-init on locale change
//   • Handles orientation detection
//
// NO business logic here. All pipeline logic lives in ARController.

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../../../core/providers/diagnosis_provider.dart';
import '../../../core/providers/language_provider.dart';
import '../controllers/ar_controller.dart';
import '../models/ar_state.dart';
import '../widgets/ar_overlay.dart';
import '../widgets/scanning_indicator.dart';

// ── Public entry point ────────────────────────────────────────────────────
class ARGuideScreen extends StatefulWidget {
  final int initialStep;
  const ARGuideScreen({required this.initialStep, super.key});

  @override
  State<ARGuideScreen> createState() => _ARGuideScreenState();
}

class _ARGuideScreenState extends State<ARGuideScreen>
    with TickerProviderStateMixin {

  // ── Animation controllers ─────────────────────────────────────────────
  late final AnimationController _toastCtrl;
  late final AnimationController _verifiedCtrl;
  late final AnimationController _spinnerCtrl;
  late final AnimationController _arrowPulseCtrl;
  late final AnimationController _bboxFadeCtrl;
  late final Animation<double>   _toastSlide;
  late final Animation<double>   _toastFade;
  late final Animation<double>   _verifiedFade;
  late final Animation<double>   _bboxFadeAnim;

  // ── Controller ────────────────────────────────────────────────────────
  late final ARController _ctrl;

  @override
  void initState() {
    super.initState();

    // ── Animations ──────────────────────────────────────────────────────
    _toastCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 340));
    _toastSlide = Tween<double>(begin: -16, end: 0).animate(
        CurvedAnimation(parent: _toastCtrl, curve: Curves.easeOutCubic));
    _toastFade = CurvedAnimation(parent: _toastCtrl, curve: Curves.easeOut);

    _verifiedCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 500));
    _verifiedFade = CurvedAnimation(parent: _verifiedCtrl, curve: Curves.easeOut);

    _spinnerCtrl = AnimationController(vsync: this,
        duration: const Duration(seconds: 1))..repeat();

    _arrowPulseCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 900))
      ..repeat(reverse: true);

    _bboxFadeCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 350));
    _bboxFadeAnim = CurvedAnimation(parent: _bboxFadeCtrl, curve: Curves.easeOut);

    // ── Controller ───────────────────────────────────────────────────────
    _ctrl = ARController(
      getContext: () => context,
      setState:   (fn) { if (mounted) setState(fn); },
      isMounted:  () => mounted,
    );
    _ctrl.currentStep    = widget.initialStep;

    // Wire animations into controller
    _ctrl.bboxFadeCtrl  = _bboxFadeCtrl;
    _ctrl.arrowPulseCtrl = _arrowPulseCtrl;
    _ctrl.verifiedCtrl  = _verifiedCtrl;
    _ctrl.toastCtrl     = _toastCtrl;
    _ctrl.spinnerCtrl   = _spinnerCtrl;
    _ctrl.bboxFadeAnim  = _bboxFadeAnim;
    _ctrl.verifiedFade  = _verifiedFade;
    _ctrl.toastSlide    = _toastSlide;
    _ctrl.toastFade     = _toastFade;

    _ctrl.initCamera();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final langCode = context.read<LanguageProvider>().languageCode;
    _ctrl.initTts(langCode);
  }

  @override
  void dispose() {
    _toastCtrl.dispose();
    _verifiedCtrl.dispose();
    _spinnerCtrl.dispose();
    _arrowPulseCtrl.dispose();
    _bboxFadeCtrl.dispose();
    _ctrl.dispose();
    super.dispose();
  }

  // ── Build ─────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final prov     = context.watch<DiagnosisProvider>();
    final steps    = prov.solution?.steps ?? demoSteps;
    final step     = _ctrl.currentStep < steps.length ? steps[_ctrl.currentStep] : null;
    final total    = steps.length;
    final nextStep = (_ctrl.currentStep + 1) < steps.length
        ? steps[_ctrl.currentStep + 1] : null;
    final screenH  = MediaQuery.of(context).size.height;
    final screenW  = MediaQuery.of(context).size.width;
    final safePad  = MediaQuery.of(context).padding;
    final isHindi  = context.watch<LanguageProvider>().languageCode == 'hi';

    // Auto-trigger panel for non-visual steps
    _ctrl.maybeShowInspectionPanel(step);

    // ── Orientation change → reset bbox ─────────────────────────────────
    final curOrientation = MediaQuery.of(context).orientation;
    if (_ctrl.lastOrientation != null &&
        _ctrl.lastOrientation != curOrientation) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) _ctrl.onOrientationChange();
      });
    }
    _ctrl.lastOrientation = curOrientation;

    // ── Geometry ─────────────────────────────────────────────────────────
    final viewportH    = screenH - kPanelHeight - safePad.top;
    const colTotal     = kBoxSize + 28 + kCaptureSize + 10 + 20.0;
    final colTopOffset = ((viewportH - colTotal) / 2).clamp(0.0, viewportH);
    const shiftDown    = 50.0;
    final boxTop       = safePad.top + colTopOffset + shiftDown;
    final boxLeft      = (screenW - kBoxSize) / 2;
    final boxRect      = Rect.fromLTWH(boxLeft, boxTop, kBoxSize, kBoxSize);

    // ── Orientation-aware preview aspect ratio ────────────────────────────
    final camVal      = _ctrl.cameraController?.value;
    final previewSize = camVal?.previewSize;
    final previewW    = screenW;
    final isLandscape = MediaQuery.of(context).orientation == Orientation.landscape;
    final double previewH;
    if (previewSize == null) {
      previewH = screenH;
    } else if (isLandscape) {
      previewH = screenW * (previewSize.height / previewSize.width);
    } else {
      previewH = screenW * (previewSize.width / previewSize.height);
    }

    final captureLabel = switch (_ctrl.arState) {
      ARState.guiding  => isHindi ? 'सत्यापित करें'     : 'Verify Part',
      ARState.locating => isHindi ? 'भाग ढूंढ रहे हैं…' : 'Locating…',
      _                => isHindi ? 'भाग विश्लेषण करें' : 'Analyze Part',
    };

    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        backgroundColor: C.bg,
        body: Stack(
          fit: StackFit.expand,
          children: [

            // 1. Camera / stub / perm-denied
            if (_ctrl.cameraPermDenied)
              CameraPermDeniedFallback(isHindi: isHindi)
            else if (_ctrl.cameraReady && _ctrl.cameraController != null)
              Positioned.fill(child: CameraPreview(_ctrl.cameraController!))
            else
              const CameraPreviewStub(),

            // 2. Blur surround
            if (!_ctrl.cameraPermDenied)
              BlurSurround(boxRect: boxRect, cornerRadius: 20),

            // 3. AR Arrow overlay
            if ((_ctrl.arState == ARState.guiding || _ctrl.bboxLocked) &&
                _ctrl.tracking.smoothBbox != null)
              ARArrowOverlay(
                bbox:      _ctrl.tracking.smoothBbox!,
                fadeAnim:  _bboxFadeAnim,
                pulseCtrl: _arrowPulseCtrl,
                partLabel: _ctrl.partDescription.isNotEmpty
                    ? _ctrl.partDescription
                    : (step?.visualCue ?? '').replaceAll('_', ' '),
                isHindi:   isHindi,
                previewW:  previewW,
                previewH:  previewH,
              ),

            // 3b. Camera guidance chip
            if (_ctrl.arState == ARState.locating &&
                _ctrl.cameraGuidance.isNotEmpty)
              Positioned(
                top: boxRect.top - 64, left: 24, right: 24,
                child: Center(
                  child: CameraGuidanceChip(
                    message: _ctrl.cameraGuidance, isHindi: isHindi),
                ),
              ),

            // 4. Gradient tint
            GradientOverlay(arState: _ctrl.arState),

            // 5. Panel scrim
            if (_ctrl.panelExpanded)
              Positioned.fill(
                child: GestureDetector(
                  onTap: () => setState(() => _ctrl.panelExpanded = false),
                  child: Container(color: Colors.black.withOpacity(0.60)),
                ),
              ),

            // 6. Centre: scan box + capture button
            IgnorePointer(
              ignoring: _ctrl.panelExpanded,
              child: AnimatedOpacity(
                duration: const Duration(milliseconds: 260),
                opacity: _ctrl.panelExpanded ? 0.0 : 1.0,
                child: CentreArea(
                  arState:      _ctrl.arState,
                  verifiedFade: _verifiedFade,
                  onCapture:    _ctrl.onCapture,
                  boxRect:      boxRect,
                  captureLabel: captureLabel,
                  isHindi:      isHindi,
                ),
              ),
            ),

            // 7. Top bar
            Positioned(
              top: safePad.top + 16, left: 20, right: 20,
              child: TopBar(
                voiceActive: _ctrl.voiceActive,
                arState:     _ctrl.arState,
                isHindi:     isHindi,
                onClose: () => Navigator.of(context).pop(),
                onVoice: () async {
                  HapticFeedback.lightImpact();
                  if (_ctrl.voiceActive) {
                    setState(() => _ctrl.voiceActive = false);
                    await _ctrl.tts.stop();
                  } else {
                    setState(() => _ctrl.voiceActive = true);
                    final stepText = step != null
                        ? step.getLocalizedText(isHindi)
                        : (isHindi ? 'घटक ढूंढें' : 'Locate the component');
                    await _ctrl.tts.stop();
                    await _ctrl.tts.speak(stepText);
                  }
                },
              ),
            ),

            // 8. Toast
            ToastPositioned(
              slideAnim:       _toastSlide,
              fadeAnim:        _toastFade,
              kind:            _ctrl.toastKind,
              spinnerCtrl:     _spinnerCtrl,
              topOffset:       safePad.top + 66,
              dynamicFeedback: _ctrl.dynamicFeedback,
              isHindi:         isHindi,
            ),

            // 9. Next Part button
            if (_ctrl.arState == ARState.verified && !_ctrl.panelExpanded)
              Positioned(
                bottom: kPanelHeight + 24, left: 24, right: 24,
                child: FadeTransition(
                  opacity: _verifiedFade,
                  child: NextPartButton(
                    onTap:   () => _ctrl.nextPart(steps),
                    isHindi: isHindi,
                  ),
                ),
              ),

            // 10. Bottom panel
            Positioned(
              left: 0, right: 0, bottom: 0,
              child: BottomPanel(
                step:        step,
                nextStep:    nextStep,
                stepIndex:   _ctrl.currentStep,
                total:       total,
                arState:     _ctrl.arState,
                expanded:    _ctrl.panelExpanded,
                screenH:     screenH,
                feedbackMsg: _ctrl.cameraGuidance.isNotEmpty
                    ? _ctrl.cameraGuidance : _ctrl.dynamicFeedback,
                isHindi:     isHindi,
                onToggle:    () => setState(
                    () => _ctrl.panelExpanded = !_ctrl.panelExpanded),
              ),
            ),

            // 11. Danger panel
            if (_ctrl.arState == ARState.danger)
              Positioned.fill(
                child: DangerPanel(
                  message:   _ctrl.dangerMessage,
                  isHindi:   isHindi,
                  onDismiss: () async {
                    await _ctrl.resumeCamera();
                    _ctrl.transitionTo(ARState.locating);
                    _ctrl.maybeStartLocateLoop();
                  },
                ),
              ),

            // 12. Inspection panel
            if (_ctrl.inspectionPanelVisible && step != null)
              Positioned(
                left: 0, right: 0, bottom: 0,
                child: InspectionPanel(
                  step:      step,
                  isHindi:   isHindi,
                  onAnswer:  (opt) => _ctrl.onInspectionAnswer(opt, steps),
                  onDone:    _ctrl.onActionDone,
                  onDismiss: () => setState(
                      () => _ctrl.inspectionPanelVisible = false),
                ),
              ),
          ],
        ),
      ),
    );
  }
}