// ignore_for_file: deprecated_member_use
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import '../../core/providers/diagnosis_provider.dart';
import '../../core/providers/language_provider.dart';
import '../../services/api_service.dart';
import 'package:flutter_tts/flutter_tts.dart';

// ═══════════════════════════════════════════════════════════════════════════
// AR Guide Screen — Production v3.1
//
// FIX v3.1:
//   Removed animated corner brackets from _SolidBoxPainter completely.
//   The brackets were drawn slightly outside the box bounds, causing the
//   "uneven rounded corner" artifact visible in the emulator preview.
//   Now _SolidBoxPainter draws ONLY the clean solid rounded-rect border.
//   _cornerCtrl / _cornerPulse animation controllers are also removed.
// ═══════════════════════════════════════════════════════════════════════════

abstract class _C {
  static const bg          = Color(0xFF1A1A1A);
  static const primary     = Color(0xFF34D399);
  static const gold        = Color(0xFFFFCC15);
  static const danger      = Color(0xFFFF4B4B);
  static const warning     = Color(0xFFF59E0B);
  static const panelBg1    = Color(0xFF1B1B1B);
  static const panelBg2    = Color(0xFF111111);
  static const toastBg     = Color(0xF01E1E1E);
  static const goldBadgeBg = Color(0xFF5A4A2A);
  static const textPrimary = Color(0xFFFFFFFF);
  static const textSoft    = Color(0xFFEAEAEA);
  static const textMuted   = Color(0xFFCFCFCF);
}

enum _ARState  { scanning, analyzing, unclear, verified, danger }
enum _ToastKind { analyzing, sent, analyzed, resultOk, resultWarn, error }

const double _kPanelHeight = 240.0;
const double _kCaptureSize = 72.0;
const double _kBoxSize     = 300.0;

// ═══════════════════════════════════════════════════════════════════════════
class ARGuideScreen extends StatefulWidget {
  final int initialStep;
  const ARGuideScreen({super.key, this.initialStep = 0});

  @override
  State<ARGuideScreen> createState() => _ARGuideScreenState();
}

class _ARGuideScreenState extends State<ARGuideScreen>
    with TickerProviderStateMixin {

  late int   _currentStep;
  _ARState   _arState       = _ARState.scanning;
  bool       _voiceActive   = false;
  bool       _panelExpanded = false;
  int        _attemptCount    = 0;
  final List<Map<String, dynamic>> _attemptResults = [];  // visual memory for current step
  _ToastKind _toastKind     = _ToastKind.analyzing;
  String     _dangerMessage = '';
  String     _dynamicFeedback = '';

  // ── Inspection / action panel state ──────────────────────────────────────
  bool   _inspectionPanelVisible = false;
  final FlutterTts _tts = FlutterTts();
  CameraController? _cameraController;
  bool _cameraReady      = false;
  bool _cameraPermDenied = false;

  Future<void> _initCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      if (mounted) setState(() => _cameraPermDenied = true);
      return;
    }
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) return;
      // medium = 1280×720. "low" (352×288) gave Gemini Vision too few pixels to
      // distinguish machine parts, causing false "blurry/unclear" rejections.
      _cameraController = CameraController(
        cameras.first, ResolutionPreset.veryHigh,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await _cameraController!.initialize();

      // Auto-focus + auto-exposure: both dramatically improve part recognition.
      // setFocusMode keeps the lens continuously hunting for sharpness so a
      // stationary phone still locks onto a part at any depth.
      // setExposureMode prevents the frame from being over- or under-exposed in
      // low-light engine bays, which is the most common false-blur trigger.
      // Wrapped in try/catch because a handful of older devices throw
      // CameraException("focusModeNotSupported") — we still want the camera.
      try {
        await _cameraController!.setFocusMode(FocusMode.auto);
        await _cameraController!.setExposureMode(ExposureMode.auto);
      } catch (modeErr) {
        debugPrint('ARGuide: focus/exposure mode not supported — $modeErr');
      }

      if (mounted) setState(() => _cameraReady = true);
    } catch (e) {
      debugPrint('ARGuide: camera init failed — $e');
    }
  }

  Future<void> _pauseCamera()  async {
    if (_cameraReady) await _cameraController?.pausePreview();
  }
  Future<void> _resumeCamera() async {
    if (_cameraReady) await _cameraController?.resumePreview();
  }
  Future<File> _captureFrame() async {
    final xFile = await _cameraController!.takePicture();
    return File(xFile.path);
  }

  late final AnimationController _toastCtrl;
  late final AnimationController _verifiedCtrl;
  late final AnimationController _spinnerCtrl;

  late final Animation<double> _toastSlide;
  late final Animation<double> _toastFade;
  late final Animation<double> _verifiedFade;

  Timer? _toastTimer;

  @override
  void initState() {
    super.initState();
    _currentStep = widget.initialStep;
    _toastCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 340));
    _toastSlide = Tween<double>(begin: -16, end: 0).animate(
        CurvedAnimation(parent: _toastCtrl, curve: Curves.easeOutCubic));
    _toastFade = CurvedAnimation(parent: _toastCtrl, curve: Curves.easeOut);

    _verifiedCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 500));
    _verifiedFade = CurvedAnimation(
        parent: _verifiedCtrl, curve: Curves.easeOut);

    // NOTE: _cornerCtrl removed — no more corner bracket animation
    _spinnerCtrl = AnimationController(
        vsync: this, duration: const Duration(seconds: 1))
      ..repeat();
    _initCamera();
    // TTS language is applied in didChangeDependencies() once the tree is ready
  }

  // Called on first frame AND whenever an InheritedWidget this widget depends
  // on changes (e.g. LanguageProvider.setLocale). Both cases need a fresh TTS
  // language — we always stop() first because flutter_tts caches the previous
  // voice engine internally and setLanguage() alone is not enough to flush it.
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final langCode = context.read<LanguageProvider>().languageCode;
    _initTts(langCode);
  }

  /// Maps app locale → BCP-47 TTS tag.
  static String _ttsLangFor(String code) {
    switch (code) {
      case 'hi': return 'hi-IN';
      case 'pa': return 'pa-IN';
      default:   return 'en-US';
    }
  }

  Future<void> _initTts(String langCode) async {
    // stop() before setLanguage() is required: flutter_tts keeps the previous
    // voice engine loaded; calling setLanguage() while it is cached does nothing
    // on several Android OEM builds.  stop() forces a full engine reset.
    await _tts.stop();
    await _tts.setLanguage(_ttsLangFor(langCode));
    await _tts.setSpeechRate(0.48);
    _tts.setCompletionHandler(() {
      if (mounted) setState(() => _voiceActive = false);
    });
    _tts.setCancelHandler(() {
      if (mounted) setState(() => _voiceActive = false);
    });
  }

  @override
  void dispose() {
    _tts.stop();
    _toastTimer?.cancel();
    _toastCtrl.dispose();
    _verifiedCtrl.dispose();
    _spinnerCtrl.dispose();
    _cameraController?.dispose();
    super.dispose();
  }

  Duration _dismissDuration(_ToastKind k) {
    switch (k) {
      case _ToastKind.analyzing:  return const Duration(seconds: 60);
      case _ToastKind.sent:       return const Duration(seconds: 5);
      case _ToastKind.analyzed:   return const Duration(seconds: 5);
      case _ToastKind.resultOk:   return const Duration(seconds: 6);
      case _ToastKind.resultWarn: return const Duration(seconds: 8);
      case _ToastKind.error:      return const Duration(seconds: 8);
    }
  }

  Future<void> _showToast(_ToastKind kind) async {
    if (!mounted) return;
    if (_toastCtrl.isAnimating || _toastCtrl.isCompleted) {
      await _toastCtrl.reverse();
    }
    if (!mounted) return;
    setState(() => _toastKind = kind);
    _toastCtrl.forward(from: 0);
    _scheduleToastDismiss(_dismissDuration(kind));
  }

  void _scheduleToastDismiss(Duration delay) {
    _toastTimer?.cancel();
    _toastTimer = Timer(delay, () async {
      if (!mounted) return;
      await _toastCtrl.reverse();
      if (mounted && _arState == _ARState.unclear) {
        _transitionTo(_ARState.scanning);
      }
    });
  }

  void _transitionTo(_ARState next) {
    if (!mounted) return;
    setState(() => _arState = next);
    switch (next) {
      case _ARState.scanning:
        _toastTimer?.cancel();
        _toastCtrl.reverse();
        break;
      case _ARState.analyzing:
        break;
      case _ARState.unclear:
        break;
      case _ARState.verified:
        _toastTimer?.cancel();
        _toastCtrl.reverse();
        _verifiedCtrl.forward(from: 0);
        break;
      case _ARState.danger:
        _toastTimer?.cancel();
        _toastCtrl.reset();
        break;
    }
  }

  Future<void> _onCapture() async {
    if (_arState != _ARState.scanning) return;
    if (!_cameraReady || _cameraController == null) return;

    // Non-visual steps use the inspection / action panel, not the camera
    final prov  = context.read<DiagnosisProvider>();
    final steps = prov.solution?.steps ?? _demoSteps;
    final step  = _currentStep < steps.length ? steps[_currentStep] : null;
    if (step != null && (step.requiresDecisionPanel || step.isActionStep)) {
      setState(() => _inspectionPanelVisible = true);
      return;
    }

    HapticFeedback.mediumImpact();
    _attemptCount++;
    await _pauseCamera();
    _transitionTo(_ARState.analyzing);
    await _showToast(_ToastKind.analyzing);

    final File imageFile;
    try {
      imageFile = await _captureFrame();
    } catch (_) {
      await _resumeCamera();
      _transitionTo(_ARState.scanning);
      return;
    }

    await _showToast(_ToastKind.sent);

    final stepText = step?.textEn.isNotEmpty == true
        ? step!.textEn : step?.text ?? '';
    final machine  = prov.solution?.machineType ?? 'tractor';
    final isHindi = context.read<LanguageProvider>().languageCode == 'hi';
    final problem  = prov.solution?.getLocalizedProblem(isHindi) ?? '';

    Map<String, dynamic> result;
    try {
      result = await ApiService.verifyStep(
        imageFile:      imageFile,
        stepText:       stepText,
        machineType:    machine,
        problemContext: problem,
        attemptCount:   _attemptCount,
        previousSteps:  jsonEncode(_attemptResults),  // send visual memory to server
      );
    } on Exception {
      await _showToast(_ToastKind.error);
      await _resumeCamera();
      _transitionTo(_ARState.scanning);
      return;
    }

    await _showToast(_ToastKind.analyzed);

    final isDangerous = result['danger'] == true ||
        result['status'] == 'danger' ||
        (result['severity'] as String? ?? '').toLowerCase() == 'critical';

    if (isDangerous) {
      if (mounted) {
        setState(() {
          _dangerMessage = result['danger_message'] as String? ??
              'STOP — Critical safety hazard detected!\n\n'
              'The machine appears to be running or there is an immediate risk.\n'
              'Do NOT proceed until the machine is fully off and safe.';
        });
      }
      _transitionTo(_ARState.danger);
      HapticFeedback.vibrate();
      return;
    }

    final verified = result['verified'] == true ||
        result['status'] == 'verified' ||
        result['correct'] == true;

    if (verified) {
      await _showToast(_ToastKind.resultOk);
      _transitionTo(_ARState.verified);
      HapticFeedback.heavyImpact();
    } else {
      // Append this failed attempt to history so next call has full context
      _attemptResults.add({
        'attempt_count': result['attempt_count'] ?? _attemptCount,
        'status':        result['status']        ?? 'unclear',
        'detected_part': result['detected_part'] ?? '',
        'feedback':      result['feedback']      ?? '',
      });

      if (mounted) {
        final isHindi = context.read<LanguageProvider>().languageCode == 'hi';
        setState(() {
          final rawFeedback = isHindi
              ? (result['feedback_hi'] ?? result['feedback'])
              : result['feedback'];
          _dynamicFeedback = rawFeedback ??
                             result['ai_observation'] ??
                             'Image unclear or wrong part captured — see hint below for guidance.';
        });
      }

      await _showToast(_ToastKind.resultWarn);
      await _resumeCamera();
      _transitionTo(_ARState.unclear);
    }
  }

  void _nextPart(List<StepData> steps) {
    if (_currentStep + 1 >= steps.length) return;
    HapticFeedback.lightImpact();
    _resumeCamera();
    setState(() {
      _currentStep++;
      _arState        = _ARState.scanning;
      _attemptCount   = 0;
      _panelExpanded  = false;
      _dynamicFeedback = '';
      _inspectionPanelVisible = false;
    });
    _attemptResults.clear();
    _verifiedCtrl.reset();

    // Auto-show inspection panel for the new step if it requires one
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final prov  = context.read<DiagnosisProvider>();
      final steps = prov.solution?.steps ?? _demoSteps;
      if (_currentStep < steps.length) {
        final s = steps[_currentStep];
        if (s.requiresDecisionPanel || s.isActionStep) {
          setState(() => _inspectionPanelVisible = true);
        }
      }
    });
  }

  // ── Inspection / observation panel ───────────────────────────────────────

  /// Farmer tapped an answer button in the inspection panel.
  void _onInspectionAnswer(StepOption option, List<StepData> steps) {
    HapticFeedback.mediumImpact();

    // Record the answer in attempt history (same structure as visual attempts)
    _attemptResults.add({
      'attempt_count': 1,
      'status':        'answered',
      'detected_part': option.id,
      'feedback':      option.labelEn,
    });

    setState(() {
      _inspectionPanelVisible = false;
    });

    // Route to next_step if provided, otherwise advance linearly
    if (option.nextStep.isNotEmpty) {
      final prov = context.read<DiagnosisProvider>();
      final targetIdx = prov.solution?.indexOfStepId(option.nextStep) ?? -1;
      if (targetIdx >= 0 && targetIdx < steps.length) {
        // Jump to the routed step
        Future.delayed(const Duration(milliseconds: 300), () {
          if (!mounted) return;
          setState(() {
            _currentStep            = targetIdx;
            _arState                = _ARState.scanning;
            _attemptCount           = 0;
            _panelExpanded          = false;
            _dynamicFeedback        = '';
            _inspectionPanelVisible = false;
          });
          _attemptResults.clear();
          _verifiedCtrl.reset();
          // Auto-show panel if next step is also inspection
          final nextS = steps[targetIdx];
          if (nextS.requiresDecisionPanel || nextS.isActionStep) {
            setState(() => _inspectionPanelVisible = true);
          }
        });
        return;
      }
    }

    // Linear advance — mark this step as verified and show Next Part button
    Future.delayed(const Duration(milliseconds: 300), () {
      if (!mounted) return;
      _transitionTo(_ARState.verified);
      HapticFeedback.heavyImpact();
    });
  }

  /// Farmer tapped "Done" on an action step.
  void _onActionDone() {
    HapticFeedback.heavyImpact();
    setState(() => _inspectionPanelVisible = false);
    _transitionTo(_ARState.verified);
  }

  /// Expose inspection panel when entering a non-visual step.
  void _maybeShowInspectionPanel(StepData? step) {
    if (step == null) return;
    if ((step.requiresDecisionPanel || step.isActionStep) &&
        !_inspectionPanelVisible &&
        _arState == _ARState.scanning) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) setState(() => _inspectionPanelVisible = true);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final prov     = context.watch<DiagnosisProvider>();
    final steps    = prov.solution?.steps ?? _demoSteps;
    final step     = _currentStep < steps.length ? steps[_currentStep] : null;
    final total    = steps.length;
    final nextStep = (_currentStep + 1) < steps.length
        ? steps[_currentStep + 1] : null;
    final screenH  = MediaQuery.of(context).size.height;
    final screenW  = MediaQuery.of(context).size.width;
    final safePad  = MediaQuery.of(context).padding;
    final isHindi  = context.watch<LanguageProvider>().languageCode == 'hi';

    // Auto-trigger panel for non-visual steps on first render
    _maybeShowInspectionPanel(step);

    // Compute the scan-box rect so the blur cut-out aligns with the painted box.
    final viewportH    = screenH - _kPanelHeight - safePad.top;
    const colTotal     = _kBoxSize + 28 + _kCaptureSize + 10 + 20.0;
    final colTopOffset = ((viewportH - colTotal) / 2).clamp(0.0, viewportH);
    // ADDED: Shift the entire AR assembly down by 50 pixels to clear the toast
    final double shiftDown = 50.0; 
    final boxTop       = safePad.top + colTopOffset + shiftDown;
    final boxLeft      = (screenW - _kBoxSize) / 2;
    final boxRect      = Rect.fromLTWH(boxLeft, boxTop, _kBoxSize, _kBoxSize);

    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        backgroundColor: _C.bg,
        body: Stack(
          fit: StackFit.expand,
          children: [

            // 1. Full-screen camera / stub / perm-denied
            if (_cameraPermDenied)
              _CameraPermDeniedFallback(isHindi: isHindi)
            else if (_cameraReady && _cameraController != null)
              Positioned.fill(child: CameraPreview(_cameraController!))
            else
              const _CameraPreviewStub(),

            // 2. Frosted blur — covers everything OUTSIDE the scan box
            if (!_cameraPermDenied)
              _BlurSurround(boxRect: boxRect, cornerRadius: 20),

            // 3. Top-bottom gradient tint
            _GradientOverlay(arState: _arState),

            // 4. Expanded panel scrim
            if (_panelExpanded)
              Positioned.fill(
                child: GestureDetector(
                  onTap: () => setState(() => _panelExpanded = false),
                  child: Container(color: Colors.black.withOpacity(0.60)),
                ),
              ),

            // 5. Centre: scan box + capture button
            // 5. Centre: scan box + capture button
            IgnorePointer(
              ignoring: _panelExpanded,
              child: AnimatedOpacity(
                duration: const Duration(milliseconds: 260),
                opacity: _panelExpanded ? 0.0 : 1.0,
                child: _CentreArea(
                  arState:      _arState,
                  verifiedFade: _verifiedFade,
                  onCapture:    _onCapture,
                  boxRect:      boxRect,
                  isHindi:      isHindi,
                ),
              ),
            ),

            // 6. Top bar
            Positioned(
              top: safePad.top + 16, 
              left: 20,
              right: 20,
              child: _TopBar(
                voiceActive: _voiceActive,
                isHindi: isHindi,
                onClose: () => Navigator.of(context).pop(),
                onVoice: () async {
                  HapticFeedback.lightImpact();
                  if (_voiceActive) {
                    setState(() => _voiceActive = false);
                    await _tts.stop();
                  } else {
                    setState(() => _voiceActive = true);
                    final stepText = step != null
                        ? step.getLocalizedText(isHindi)
                        : (isHindi ? 'घटक ढूंढें' : 'Locate the component');
                    await _tts.stop();
                    await _tts.speak(stepText);
                  }
                },
              ),
            ),

            // 7. Toast — below top bar, full text
            _ToastPositioned(
              slideAnim:   _toastSlide,
              fadeAnim:    _toastFade,
              kind:        _toastKind,
              spinnerCtrl: _spinnerCtrl,
              topOffset:   safePad.top + 66,
              dynamicFeedback: _dynamicFeedback,
              isHindi:     isHindi,
            ),

            // 8. Next Part button (verified only)
            if (_arState == _ARState.verified && !_panelExpanded)
              Positioned(
                bottom: _kPanelHeight + 24,
                left: 24, right: 24,
                child: FadeTransition(
                  opacity: _verifiedFade,
                  child: _NextPartButton(
                    onTap: () => _nextPart(steps),
                    isHindi: isHindi,
                  ),
                ),
              ),

            // 9. Bottom panel
            Positioned(
              left: 0, right: 0, bottom: 0,
              child: _BottomPanel(
                step:        step,
                nextStep:    nextStep,
                stepIndex:   _currentStep,
                total:       total,
                arState:     _arState,
                expanded:    _panelExpanded,
                screenH:     screenH,
                feedbackMsg: _dynamicFeedback,
                isHindi:     isHindi,
                onToggle:    () => setState(
                    () => _panelExpanded = !_panelExpanded),
              ),
            ),

            // 10. Danger panel — topmost
            if (_arState == _ARState.danger)
              Positioned.fill(
                child: _DangerPanel(
                  message: _dangerMessage,
                  isHindi: isHindi,
                  onDismiss: () async {
                    await _resumeCamera();
                    _transitionTo(_ARState.scanning);
                  },
                ),
              ),

            // 11. Inspection / action / observation panel
            //     Slides up from the bottom on non-visual steps.
            if (_inspectionPanelVisible && step != null)
              Positioned(
                left: 0, right: 0, bottom: 0,
                child: _InspectionPanel(
                  step:       step,
                  isHindi:    isHindi,
                  onAnswer:   (opt) => _onInspectionAnswer(opt, steps),
                  onDone:     _onActionDone,
                  onDismiss:  () => setState(() => _inspectionPanelVisible = false),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Blur surround — blurs everything OUTSIDE the scan box
// ─────────────────────────────────────────────────────────────────────────
class _BlurSurround extends StatelessWidget {
  final Rect   boxRect;
  final double cornerRadius;
  const _BlurSurround({required this.boxRect, required this.cornerRadius});

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: ClipPath(
        clipper: _SurroundClipper(
            boxRect: boxRect, cornerRadius: cornerRadius),
        child: BackdropFilter(
          filter: ui.ImageFilter.blur(sigmaX: 6, sigmaY: 6),
          child: Container(color: Colors.black.withOpacity(0.45)),
        ),
      ),
    );
  }
}

class _SurroundClipper extends CustomClipper<Path> {
  final Rect   boxRect;
  final double cornerRadius;
  const _SurroundClipper(
      {required this.boxRect, required this.cornerRadius});

  @override
  Path getClip(Size size) {
    final outer = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height));
    final hole = Path()
      ..addRRect(RRect.fromRectAndRadius(
          boxRect, Radius.circular(cornerRadius)));
    return Path.combine(PathOperation.difference, outer, hole);
  }

  @override
  bool shouldReclip(_SurroundClipper old) =>
      old.boxRect != boxRect || old.cornerRadius != cornerRadius;
}

// ─────────────────────────────────────────────────────────────────────────
// Camera stubs / fallbacks
// ─────────────────────────────────────────────────────────────────────────
class _CameraPreviewStub extends StatelessWidget {
  const _CameraPreviewStub();
  @override
  Widget build(BuildContext context) => Container(
    decoration: const BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topCenter, end: Alignment.bottomCenter,
        colors: [Color(0xFF2A2A2A), Color(0xFF1A1A1A)],
      ),
    ),
  );
}

class _CameraPermDeniedFallback extends StatelessWidget {
  final bool isHindi;
  const _CameraPermDeniedFallback({required this.isHindi});
  @override
  Widget build(BuildContext context) {
    return Container(
      color: _C.bg,
      child: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 40),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.no_photography_rounded,
                  color: _C.textMuted, size: 56),
              const SizedBox(height: 20),
              Text(isHindi ? 'कैमरा एक्सेस आवश्यक है' : 'Camera Access Required',
                textAlign: TextAlign.center,
                style: GoogleFonts.inter(fontSize: 20,
                    fontWeight: FontWeight.w700, color: _C.textPrimary)),
              const SizedBox(height: 10),
              Text(
                isHindi
                    ? 'AR गाइड उपयोग करने के लिए डिवाइस सेटिंग में कैमरा एक्सेस दें।'
                    : 'Please allow camera access in your device settings to use the AR guide.',
                textAlign: TextAlign.center,
                style: GoogleFonts.inter(
                    fontSize: 14, color: _C.textMuted, height: 1.55)),
              const SizedBox(height: 28),
              GestureDetector(
                onTap: openAppSettings,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 32, vertical: 14),
                  decoration: BoxDecoration(
                    color: _C.primary,
                    borderRadius: BorderRadius.circular(16)),
                  child: Text(isHindi ? 'सेटिंग खोलें' : 'Open Settings',
                    style: GoogleFonts.inter(fontSize: 15,
                        fontWeight: FontWeight.w600, color: Colors.black87)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Gradient overlay
// ─────────────────────────────────────────────────────────────────────────
class _GradientOverlay extends StatelessWidget {
  final _ARState arState;
  const _GradientOverlay({required this.arState});

  @override
  Widget build(BuildContext context) {
    final isVerified = arState == _ARState.verified;
    return AnimatedContainer(
      duration: const Duration(milliseconds: 400),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end:   Alignment.bottomCenter,
          stops: const [0.0, 0.30, 0.70, 1.0],
          colors: isVerified
              ? [
                  const Color(0xFF1A3A2A).withOpacity(0.80),
                  Colors.transparent,
                  Colors.transparent,
                  const Color(0xFF0D2016).withOpacity(0.92),
                ]
              : [
                  Colors.black.withOpacity(0.60),
                  Colors.transparent,
                  Colors.transparent,
                  Colors.black.withOpacity(0.72),
                ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Top Bar — ✕ left · speaking pill CENTER · volume right
// ─────────────────────────────────────────────────────────────────────────
class _TopBar extends StatelessWidget {
  final bool         voiceActive;
  final bool         isHindi;
  final VoidCallback onClose;
  final VoidCallback onVoice;
  const _TopBar({
    required this.voiceActive,
    required this.isHindi,
    required this.onClose,
    required this.onVoice,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        // Left (Close) and Right (Volume) Buttons
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            _CircleBtn(icon: Icons.close_rounded, onTap: onClose),
            _CircleBtn(
              icon: voiceActive
                  ? Icons.volume_up_rounded
                  : Icons.volume_off_rounded,
              onTap: onVoice,
              activeDot: voiceActive,
            ),
          ],
        ),
        
        // Centered Speaking Pill
        if (voiceActive) 
          _SpeakingPill(isHindi: isHindi),
      ],
    );
  }
}

class _CircleBtn extends StatelessWidget {
  final IconData     icon;
  final VoidCallback onTap;
  final bool         activeDot;
  const _CircleBtn({
    required this.icon,
    required this.onTap,
    this.activeDot = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Container(
            width: 44, height: 44,
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.50),
              shape: BoxShape.circle,
              border: Border.all(
                  color: Colors.white.withOpacity(0.12), width: 1)),
            child: Icon(icon, color: Colors.white, size: 20),
          ),
          if (activeDot)
            Positioned(
              right: 1, bottom: 1,
              child: Container(
                width: 8, height: 8,
                decoration: const BoxDecoration(
                    color: _C.primary, shape: BoxShape.circle),
              ),
            ),
        ],
      ),
    );
  }
}

class _SpeakingPill extends StatelessWidget {
  final bool isHindi;
  const _SpeakingPill({required this.isHindi});
  @override
  Widget build(BuildContext context) {
    final label = isHindi ? 'बोल रहा है...' : 'Speaking...';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.60),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: _C.primary.withOpacity(0.35), width: 1)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.graphic_eq_rounded, color: _C.primary, size: 16),
          const SizedBox(width: 6),
          Text(label,
            style: GoogleFonts.inter(
              fontSize: 12, fontWeight: FontWeight.w500,
              color: _C.textSoft)),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Toast — full text, no ellipsis, longer dismiss, below top bar
// ─────────────────────────────────────────────────────────────────────────
class _ToastPositioned extends StatelessWidget {
  final Animation<double>   slideAnim;
  final Animation<double>   fadeAnim;
  final _ToastKind          kind;
  final AnimationController spinnerCtrl;
  final double              topOffset;
  final String              dynamicFeedback;
  final bool                isHindi;

  const _ToastPositioned({
    required this.slideAnim,
    required this.fadeAnim,
    required this.kind,
    required this.spinnerCtrl,
    required this.topOffset,
    this.dynamicFeedback = '',
    this.isHindi = false,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: topOffset, left: 16, right: 16,
      child: AnimatedBuilder(
        animation: Listenable.merge([slideAnim, fadeAnim]),
        builder: (_, child) => Opacity(
          opacity: fadeAnim.value,
          child: Transform.translate(
              offset: Offset(0, slideAnim.value), child: child),
        ),
        child: _ToastCard(kind: kind, spinnerCtrl: spinnerCtrl, dynamicFeedback: dynamicFeedback, isHindi: isHindi),
      ),
    );
  }
}

class _ToastCard extends StatelessWidget {
  final _ToastKind          kind;
  final AnimationController spinnerCtrl;
  final String              dynamicFeedback;
  final bool                isHindi;
  const _ToastCard({required this.kind, required this.spinnerCtrl, this.dynamicFeedback = '', this.isHindi = false});

  @override
  Widget build(BuildContext context) {
    final Color  accent;
    final Widget leadingIcon;
    final String message;

    switch (kind) {
      case _ToastKind.analyzing:
        accent = _C.primary;
        leadingIcon = SizedBox(
          width: 22, height: 22,
          child: AnimatedBuilder(
            animation: spinnerCtrl,
            builder: (_, __) => Transform.rotate(
              angle: spinnerCtrl.value * 2 * math.pi,
              child: const CircularProgressIndicator(
                strokeWidth: 2.5,
                valueColor: AlwaysStoppedAnimation(_C.primary)),
            ),
          ),
        );
        message = isHindi
            ? 'AI को विश्लेषण के लिए छवि भेज रहा है…'
            : 'Sending image to AI for analysis…';
        break;
      case _ToastKind.sent:
        accent      = _C.primary;
        leadingIcon = const Icon(Icons.check_circle_rounded,
            color: _C.primary, size: 22);
        message = isHindi
            ? 'छवि भेजी गई — AI प्रतिक्रिया की प्रतीक्षा है…'
            : 'Image sent successfully — awaiting AI response…';
        break;
      case _ToastKind.analyzed:
        accent      = _C.primary;
        leadingIcon = const Icon(Icons.check_circle_rounded,
            color: _C.primary, size: 22);
        message = isHindi
            ? 'AI ने छवि का सफलतापूर्वक विश्लेषण किया'
            : 'Image analyzed successfully by AI';
        break;
      case _ToastKind.resultOk:
        accent      = _C.primary;
        leadingIcon = const Icon(Icons.verified_rounded,
            color: _C.primary, size: 22);
        message = isHindi
            ? 'सही भाग पहचाना गया — कोई खराबी नहीं ✓'
            : 'Correct part identified — no damage detected ✓';
        break;
      case _ToastKind.resultWarn:
        accent      = _C.warning;
        leadingIcon = const Icon(Icons.warning_amber_rounded,
            color: _C.warning, size: 22);
        message = dynamicFeedback.isNotEmpty
            ? dynamicFeedback
            : (isHindi
                ? 'छवि अस्पष्ट है — नीचे संकेत देखें'
                : 'Image unclear or wrong part captured — see hint below for guidance');
        break;
      case _ToastKind.error:
        accent      = _C.danger;
        leadingIcon = const Icon(Icons.error_outline_rounded,
            color: _C.danger, size: 22);
        message = isHindi
            ? 'कनेक्शन त्रुटि — इंटरनेट जांचें और दोबारा प्रयास करें'
            : 'Connection error — please check your internet and try again';
        break;
    }

    return Container(
      decoration: BoxDecoration(
        color: _C.toastBg,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: accent.withOpacity(0.28), width: 1),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.50),
            blurRadius: 20, offset: const Offset(0, 5)),
        ],
      ),
      child: IntrinsicHeight(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              width: 5,
              decoration: BoxDecoration(
                color: accent,
                borderRadius: const BorderRadius.only(
                  topLeft:    Radius.circular(16),
                  bottomLeft: Radius.circular(16))),
            ),
            const SizedBox(width: 12),
            Center(
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 14),
                child: leadingIcon,
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(
                    vertical: 14, horizontal: 2),
                child: Text(
                  message,
                  softWrap: true,
                  overflow: TextOverflow.visible,
                  style: GoogleFonts.inter(
                    fontSize: 14, fontWeight: FontWeight.w500,
                    color: _C.textSoft,
                    height: 1.45, letterSpacing: 0.1),
                ),
              ),
            ),
            const SizedBox(width: 12),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Centre Area — Locked to absolute coordinates to prevent double-box
// ─────────────────────────────────────────────────────────────────────────
class _CentreArea extends StatelessWidget {
  final _ARState          arState;
  final Animation<double> verifiedFade;
  final VoidCallback      onCapture;
  final Rect              boxRect;
  final bool              isHindi;

  const _CentreArea({
    required this.arState,
    required this.verifiedFade,
    required this.onCapture,
    required this.boxRect,
    required this.isHindi,
  });

  @override
  Widget build(BuildContext context) {
    final lblTarget  = isHindi ? 'लक्ष्य'      : 'TARGET';
    final lblAnalyze = isHindi ? 'भाग विश्लेषण करें' : 'Analyze Part';
    return Stack(
      children: [
        // 1. Verified Badge (Floats just above the box)
        if (arState == _ARState.verified)
          Positioned(
            top: boxRect.top - 54, 
            left: 0, right: 0,
            child: FadeTransition(
              opacity: verifiedFade,
              child: Center(child: _ComponentVerifiedBadge(isHindi: isHindi)),
            ),
          ),

        // 2. The Scan Box (Locked EXACTLY to the blur hole coordinates)
        Positioned(
          top: boxRect.top,
          left: boxRect.left,
          child: arState == _ARState.verified
              ? FadeTransition(
                  opacity: verifiedFade,
                  child: const _SolidScanBox(color: _C.primary),
                )
              : _SolidScanBox(
                  color: arState == _ARState.unclear ? _C.warning : Colors.white,
                  label: arState == _ARState.scanning ? lblTarget : null,
                ),
        ),

        // 3. The Capture Button (Positioned safely below the box)
        if (arState != _ARState.verified && arState != _ARState.danger)
          Positioned(
            top: boxRect.bottom + 28,
            left: 0, right: 0,
            child: Column(
              children: [
                _CaptureButton(
                  onTap:   onCapture,
                  enabled: arState == _ARState.scanning,
                ),
                const SizedBox(height: 10),
                AnimatedDefaultTextStyle(
                  duration: const Duration(milliseconds: 200),
                  style: GoogleFonts.inter(
                    fontSize: 17, fontWeight: FontWeight.w400,
                    color: arState == _ARState.scanning
                        ? _C.textSoft.withOpacity(0.85)
                        : _C.textMuted.withOpacity(0.35)),
                  child: Text(lblAnalyze),
                ),
              ],
            ),
          ),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Solid scan box — clean border only, label floating above
// ─────────────────────────────────────────────────────────────────────────
class _SolidScanBox extends StatelessWidget {
  final Color   color;
  final String? label;
  const _SolidScanBox({required this.color, this.label});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: _kBoxSize, height: _kBoxSize,
      child: Stack(
        clipBehavior: Clip.none, // CRITICAL: Allows the label to float outside the bounds
        alignment: Alignment.topCenter,
        children: [
          CustomPaint(
            size: const Size(_kBoxSize, _kBoxSize),
            painter: _SolidBoxPainter(color: color),
          ),
          if (label != null)
            Positioned(
              top: -38, // Pushes the label exactly above the top border
              child: Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 14, vertical: 5),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.52),
                  borderRadius: BorderRadius.circular(20)),
                child: Text(label!,
                  style: GoogleFonts.inter(
                    fontSize: 11, fontWeight: FontWeight.w700,
                    letterSpacing: 1.4,
                    color: Colors.white.withOpacity(0.82))),
              ),
            ),
        ],
      ),
    );
  }
}

// FIX v3.1: only the solid border — corner bracket code fully removed
class _SolidBoxPainter extends CustomPainter {
  final Color color;
  const _SolidBoxPainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    const double radius  = 20.0;
    const double strokeW = 2.5;

    final rect  = Rect.fromLTWH(0, 0, size.width, size.height);
    final rrect = RRect.fromRectAndRadius(rect, const Radius.circular(radius));

    // Clean solid border — nothing else
    canvas.drawRRect(rrect, Paint()
      ..color       = color.withOpacity(0.70)
      ..style       = PaintingStyle.stroke
      ..strokeWidth = strokeW);
  }

  @override
  bool shouldRepaint(_SolidBoxPainter old) => old.color != color;
}

class _ComponentVerifiedBadge extends StatelessWidget {
  final bool isHindi;
  const _ComponentVerifiedBadge({required this.isHindi});
  @override
  Widget build(BuildContext context) {
    final label = isHindi ? 'घटक सत्यापित' : 'COMPONENT VERIFIED';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.60),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _C.primary.withOpacity(0.30), width: 1)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.verified_rounded, color: _C.primary, size: 18),
          const SizedBox(width: 8),
          Text(label,
            style: GoogleFonts.inter(
              fontSize: 12, fontWeight: FontWeight.w700,
              letterSpacing: 1.0, color: _C.primary)),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Capture Button
// ─────────────────────────────────────────────────────────────────────────
class _CaptureButton extends StatefulWidget {
  final VoidCallback onTap;
  final bool         enabled;
  const _CaptureButton({required this.onTap, required this.enabled});
  @override
  State<_CaptureButton> createState() => _CaptureButtonState();
}

class _CaptureButtonState extends State<_CaptureButton> {
  bool _pressed = false;
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown:   widget.enabled ? (_) => setState(() => _pressed = true)  : null,
      onTapUp:     widget.enabled ? (_) { setState(() => _pressed = false); widget.onTap(); } : null,
      onTapCancel: widget.enabled ? () => setState(() => _pressed = false)  : null,
      child: AnimatedScale(
        scale: _pressed ? 0.94 : 1.0,
        duration: const Duration(milliseconds: 80),
        child: AnimatedOpacity(
          duration: const Duration(milliseconds: 200),
          opacity: widget.enabled ? 1.0 : 0.35,
          child: Container(
            width: _kCaptureSize, height: _kCaptureSize,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white, width: 3),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.35),
                  blurRadius: 16, offset: const Offset(0, 6)),
              ],
            ),
            child: Center(
              child: Container(
                width: _kCaptureSize - 14, height: _kCaptureSize - 14,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white.withOpacity(
                      widget.enabled ? 0.22 : 0.08)),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Next Part Button
// ─────────────────────────────────────────────────────────────────────────
class _NextPartButton extends StatefulWidget {
  final VoidCallback onTap;
  final bool isHindi;
  const _NextPartButton({required this.onTap, required this.isHindi});
  @override
  State<_NextPartButton> createState() => _NextPartButtonState();
}

class _NextPartButtonState extends State<_NextPartButton> {
  bool _pressed = false;
  @override
  Widget build(BuildContext context) {
    final label = widget.isHindi ? 'अगला भाग' : 'Next Part';
    return GestureDetector(
      onTapDown:   (_) => setState(() => _pressed = true),
      onTapUp:     (_) { setState(() => _pressed = false); widget.onTap(); },
      onTapCancel: () => setState(() => _pressed = false),
      child: AnimatedScale(
        scale: _pressed ? 0.97 : 1.0,
        duration: const Duration(milliseconds: 90),
        child: Container(
          height: 56,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(28),
            gradient: const LinearGradient(
                colors: [Color(0xFF22C55E), _C.primary]),
            boxShadow: [
              BoxShadow(
                color: _C.primary.withOpacity(0.40),
                blurRadius: 24, offset: const Offset(0, 10)),
            ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(label,
                style: GoogleFonts.inter(
                  fontSize: 17, fontWeight: FontWeight.w600,
                  color: Colors.black87)),
              const SizedBox(width: 8),
              const Icon(Icons.chevron_right_rounded,
                  color: Colors.black87, size: 22),
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Bottom Panel
// ─────────────────────────────────────────────────────────────────────────
class _BottomPanel extends StatelessWidget {
  final StepData?    step;
  final StepData?    nextStep;
  final int          stepIndex;
  final int          total;
  final _ARState     arState;
  final bool         expanded;
  final double       screenH;
  final String       feedbackMsg;
  final VoidCallback onToggle;

  final bool         isHindi;

  const _BottomPanel({
    required this.step,        required this.nextStep,
    required this.stepIndex,   required this.total,
    required this.arState,     required this.expanded,
    required this.screenH,     required this.onToggle,
    this.feedbackMsg = '',
    this.isHindi = false,
  });

  @override
  Widget build(BuildContext context) {
    final isVerified = arState == _ARState.verified;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 340),
      curve: Curves.easeOutCubic,
      constraints: BoxConstraints(
        minHeight: _kPanelHeight,
        maxHeight: expanded ? screenH * 0.60 : _kPanelHeight,
      ),
      decoration: BoxDecoration(
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        gradient: const LinearGradient(
          begin: Alignment.topCenter, end: Alignment.bottomCenter,
          colors: [_C.panelBg1, _C.panelBg2]),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.60),
            blurRadius: 40, offset: const Offset(0, -10)),
        ],
      ),
      child: SingleChildScrollView(
        physics: expanded
            ? const BouncingScrollPhysics()
            : const NeverScrollableScrollPhysics(),
        child: Padding(
          padding: EdgeInsets.fromLTRB(20, 14, 20,
              MediaQuery.of(context).padding.bottom + 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 36, height: 4,
                  margin: const EdgeInsets.only(bottom: 14),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.16),
                    borderRadius: BorderRadius.circular(2)),
                ),
              ),

              // ── When AI returns fail/unclear: replace the step header with
              //    a full-width AI correction alert. The "STEP N · PART NAME"
              //    row is hidden — the farmer's immediate next action is what
              //    matters, not the step label they already read.
              //    When verified or scanning: show the normal badge + heading.
              if (arState == _ARState.unclear && feedbackMsg.isNotEmpty) ...[
                // ── AI Correction Alert ──────────────────────────────────────
                _AIFeedbackAlert(
                  feedbackMsg: feedbackMsg,
                  stepIndex:   stepIndex,
                  step:        step,
                  isHindi:     isHindi,
                ),
              ] else ...[
                // ── Normal: badge + component label ─────────────────────────
                Row(children: [
                  _StepBadge(stepIndex: stepIndex, verified: isVerified),
                  const SizedBox(width: 10),
                  if (!isVerified)
                    Flexible(
                      child: Text(
                        (step?.visualCue ?? 'COMPONENT')
                            .toUpperCase().replaceAll('_', ' '),
                        style: GoogleFonts.inter(
                          fontSize: 12, fontWeight: FontWeight.w600,
                          letterSpacing: 1.0, color: _C.textMuted),
                      ),
                    )
                  else ...[
                    Text(isHindi ? 'चरण ${stepIndex + 1}: पूर्ण' : 'STEP ${stepIndex + 1}: COMPLETE',
                      style: GoogleFonts.inter(
                        fontSize: 12, fontWeight: FontWeight.w700,
                        letterSpacing: 0.8, color: _C.primary)),
                    const Spacer(),
                    if (total > 0)
                      Text('${stepIndex + 1} / $total',
                        style: GoogleFonts.inter(
                          fontSize: 12, color: _C.textMuted)),
                  ],
                ]),

                const SizedBox(height: 10),

                if (isVerified) ...[
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: _C.primary.withOpacity(0.14),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                          color: _C.primary.withOpacity(0.30), width: 1)),
                    child: Text(isHindi ? 'घटक सत्यापित — कोई क्षति नहीं' : 'Component Verified — No Damage Detected',
                      style: GoogleFonts.inter(
                        fontSize: 12, fontWeight: FontWeight.w600,
                        color: _C.primary)),
                  ),
                  const SizedBox(height: 10),
                ],

                GestureDetector(
                  onTap: onToggle,
                  child: _InstructionHeading(
                    step: step, nextStep: nextStep, verified: isVerified),
                ),
              ],

              if (expanded && step != null) ...[
                const SizedBox(height: 12),
                Text(
                  step!.getLocalizedText(
                    context.watch<LanguageProvider>().languageCode == 'hi'),
                  style: GoogleFonts.inter(
                    fontSize: 14, color: _C.textMuted, height: 1.55)),
                if (step!.safetyWarning != null) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: _C.warning.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                          color: _C.warning.withOpacity(0.35), width: 1)),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Padding(
                          padding: EdgeInsets.only(top: 2),
                          child: Icon(Icons.warning_amber_rounded,
                              color: _C.warning, size: 16)),
                        const SizedBox(width: 8),
                        Expanded(child: Text(step!.safetyWarning!,
                          style: GoogleFonts.inter(
                            fontSize: 13, color: _C.warning,
                            fontWeight: FontWeight.w500, height: 1.45))),
                      ],
                    ),
                  ),
                ],
              ],

              if (isVerified) ...[
                const SizedBox(height: 14),
                _ProgressBar(current: stepIndex + 1, total: total),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Danger Panel
// ─────────────────────────────────────────────────────────────────────────
class _DangerPanel extends StatelessWidget {
  final String       message;
  final bool         isHindi;
  final VoidCallback onDismiss;
  const _DangerPanel({required this.message, required this.isHindi, required this.onDismiss});

  @override
  Widget build(BuildContext context) {
    final lblDanger  = isHindi ? '⚠ खतरा पता चला' : '⚠ DANGER DETECTED';
    final lblDismiss = isHindi ? 'समझ गया — सावधानी से जारी रखें' : 'I understand — Resume safely';
    return Container(
      color: const Color(0xF2180000),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.dangerous_rounded,
                  color: Color(0xFFFF4B4B), size: 80),
              const SizedBox(height: 24),
              Text(lblDanger,
                style: GoogleFonts.inter(
                  fontSize: 26, fontWeight: FontWeight.w800,
                  color: const Color(0xFFFF4B4B), letterSpacing: 1.2)),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: const Color(0x30FF4B4B),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                      color: const Color(0xFFFF4B4B).withOpacity(0.50),
                      width: 1.5)),
                child: Text(message,
                  textAlign: TextAlign.center,
                  style: GoogleFonts.inter(
                    fontSize: 16, fontWeight: FontWeight.w500,
                    color: Colors.white, height: 1.6)),
              ),
              const SizedBox(height: 36),
              GestureDetector(
                onTap: onDismiss,
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  decoration: BoxDecoration(
                    color: const Color(0xFF2A2A2A),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                        color: Colors.white.withOpacity(0.15))),
                  child: Text(lblDismiss,
                    textAlign: TextAlign.center,
                    style: GoogleFonts.inter(
                      fontSize: 15, fontWeight: FontWeight.w600,
                      color: Colors.white)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// AI Feedback Alert — replaces the step heading when state is unclear/fail.
//
// The backend sends feedback as two lines separated by \n:
//   Line 1: what the farmer did wrong   (shown as bold header)
//   Line 2: the exact action to perform (shown as the instruction body)
//
// When only one line is returned, it fills the full body with no sub-header.
// ─────────────────────────────────────────────────────────────────────────
class _AIFeedbackAlert extends StatelessWidget {
  final String    feedbackMsg;
  final int       stepIndex;
  final StepData? step;
  final bool      isHindi;

  const _AIFeedbackAlert({
    required this.feedbackMsg,
    required this.stepIndex,
    this.step,
    required this.isHindi,
  });

  @override
  Widget build(BuildContext context) {
    final lines   = feedbackMsg.split('\n').where((l) => l.trim().isNotEmpty).toList();
    final problem = lines.isNotEmpty ? lines.first.trim() : '';
    final action  = lines.length > 1  ? lines.sublist(1).join(' ').trim() : '';

    // Original step goal — shown as a muted reminder at the bottom of the alert
    // so the farmer never loses track of what they were supposed to do.
    final goalLabel = step?.visualCue?.toUpperCase().replaceAll('_', ' ') ?? '';
    final goalText  = step != null ? step!.getLocalizedText(isHindi) : '';

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color:        _C.warning.withOpacity(0.10),
        borderRadius: BorderRadius.circular(16),
        border:       Border.all(color: _C.warning.withOpacity(0.45), width: 1.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Header bar ───────────────────────────────────────────────────
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: _C.warning.withOpacity(0.14),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(14)),
            ),
            child: Row(
              children: [
                const Icon(Icons.warning_amber_rounded, color: _C.warning, size: 16),
                const SizedBox(width: 8),
                Text(
                  isHindi
                      ? 'चरण ${stepIndex + 1} — सुधार'
                      : 'STEP ${stepIndex + 1} — CORRECTION',
                  style: GoogleFonts.inter(
                    fontSize: 11, fontWeight: FontWeight.w700,
                    letterSpacing: 0.9, color: _C.warning),
                ),
              ],
            ),
          ),

          // ── Line 1: what went wrong ──────────────────────────────────────
          if (problem.isNotEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 12, 14, 0),
              child: Text(problem,
                style: GoogleFonts.inter(
                  fontSize: 15, fontWeight: FontWeight.w700,
                  color: _C.textPrimary, height: 1.35)),
            ),

          // ── Line 2: exact action to take ────────────────────────────────
          if (action.isNotEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 6, 14, 0),
              child: Text(action,
                style: GoogleFonts.inter(
                  fontSize: 14, fontWeight: FontWeight.w500,
                  color: _C.textSoft, height: 1.5)),
            ),

          // ── Divider ──────────────────────────────────────────────────────
          if (goalText.isNotEmpty) ...[
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 12, 14, 0),
              child: Divider(
                color: Colors.white.withOpacity(0.08), height: 1),
            ),

            // ── Original goal reminder ────────────────────────────────────
            // Muted so it doesn't compete with the correction, but always
            // visible so the farmer remembers what they were trying to find.
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 10, 14, 14),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.my_location_rounded,
                      color: _C.textMuted.withOpacity(0.55), size: 13),
                  const SizedBox(width: 7),
                  Expanded(
                    child: RichText(
                      text: TextSpan(
                        style: GoogleFonts.inter(
                          fontSize: 12, color: _C.textMuted,
                          height: 1.45),
                        children: [
                          if (goalLabel.isNotEmpty) ...[
                            TextSpan(
                              text: '$goalLabel  ',
                              style: GoogleFonts.inter(
                                fontSize: 11, fontWeight: FontWeight.w700,
                                letterSpacing: 0.7,
                                color: _C.textMuted.withOpacity(0.70)),
                            ),
                          ],
                          TextSpan(
                            text: goalText,
                            style: GoogleFonts.inter(
                              fontSize: 12,
                              color: _C.textMuted.withOpacity(0.65),
                              height: 1.45),
                          ),
                        ],
                      ),
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
            ),
          ] else
            const SizedBox(height: 14),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Step Badge · Instruction Heading · Progress Bar
// ─────────────────────────────────────────────────────────────────────────
class _StepBadge extends StatelessWidget {
  final int  stepIndex;
  final bool verified;
  const _StepBadge({required this.stepIndex, required this.verified});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: verified ? _C.primary.withOpacity(0.16) : _C.goldBadgeBg,
        borderRadius: BorderRadius.circular(20)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (verified) ...[
            const Icon(Icons.check_circle_rounded,
                color: _C.primary, size: 13),
            const SizedBox(width: 5),
          ],
          Text('STEP ${stepIndex + 1}',
            style: GoogleFonts.inter(
              fontSize: 12, fontWeight: FontWeight.w700,
              letterSpacing: 1.0,
              color: verified ? _C.primary : _C.gold)),
        ],
      ),
    );
  }
}

class _InstructionHeading extends StatelessWidget {
  final StepData? step;
  final StepData? nextStep;
  final bool      verified;
  const _InstructionHeading({
    required this.step, required this.nextStep, required this.verified});

  @override
  Widget build(BuildContext context) {
    final isHindi = context.watch<LanguageProvider>().languageCode == 'hi';
    if (step == null) {
      return Text(isHindi ? 'घटक ढूंढें' : 'Locate the component',
        maxLines: 4,
        overflow: TextOverflow.ellipsis,
        style: GoogleFonts.inter(
          fontSize: 19, fontWeight: FontWeight.w700,
          color: _C.textPrimary, height: 1.3));
    }
    
    final full    = verified ? _nextInstruction(isHindi) : _currentInstruction(isHindi);
    final keyword = verified ? _nextKeyword() : _currentKeyword();

    if (keyword.isEmpty || !full.contains(keyword)) {
      return Text(full,
        maxLines: 4,
        overflow: TextOverflow.ellipsis,
        style: GoogleFonts.inter(
          fontSize: 19, fontWeight: FontWeight.w700, // Adjusted font size
          color: _C.textPrimary, height: 1.3));
    }
    
    final parts = full.split(keyword);
    return RichText(
      maxLines: 4,
      overflow: TextOverflow.ellipsis,
      text: TextSpan(
        style: GoogleFonts.inter(
          fontSize: 19, fontWeight: FontWeight.w700, // Adjusted font size
          color: _C.textPrimary, height: 1.3),
        children: [
          TextSpan(text: parts.first),
          TextSpan(text: keyword,
            style: TextStyle(
              color:           verified ? _C.primary : _C.danger,
              decoration:      TextDecoration.underline,
              decorationColor: verified ? _C.primary : _C.danger)),
          if (parts.length > 1)
            TextSpan(text: parts.sublist(1).join(keyword)),
        ],
      ),
    );
  }

  // Removed the hard 80-character substring cutoff. Flutter handles it now!
  String _currentInstruction(bool isHindi) {
    return step!.getLocalizedText(isHindi);
  }

  String _nextInstruction(bool isHindi) {
    if (nextStep == null) return isHindi ? 'सभी चरण पूर्ण!' : 'All steps complete!';
    final vc   = nextStep!.visualCue ?? '';
    final body = nextStep!.getLocalizedText(isHindi);
    if (vc.isNotEmpty) {
      final partName = vc.replaceAll('_', ' ');
      return isHindi
          ? 'शानदार। अब $partName ढूंढें।'
          : 'Excellent. Now locate the $partName.';
    }
    return isHindi ? 'शानदार। $body' : 'Excellent. $body';
  }

  String _currentKeyword() => (step!.visualCue ?? '').replaceAll('_', ' ');
  String _nextKeyword()     => (nextStep?.visualCue ?? '').replaceAll('_', ' ');
}

class _ProgressBar extends StatelessWidget {
  final int current;
  final int total;
  const _ProgressBar({required this.current, required this.total});

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(4),
      child: LinearProgressIndicator(
        value: total > 0 ? current / total : 0.0,
        minHeight: 4,
        backgroundColor: Colors.white.withOpacity(0.10),
        valueColor: const AlwaysStoppedAnimation(_C.primary),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Inspection / Action / Observation Panel
//
// Slides up from the bottom whenever step_type is:
//   • inspection  — question + answer buttons
//   • action      — instruction + Done button
//   • observation — symptom buttons
//
// The farmer never has to type anything. Buttons are large (min 56dp)
// with destructive answers coloured amber/red.
// ─────────────────────────────────────────────────────────────────────────
class _InspectionPanel extends StatefulWidget {
  final StepData  step;
  final bool      isHindi;
  final void Function(StepOption) onAnswer;
  final VoidCallback               onDone;
  final VoidCallback               onDismiss;

  const _InspectionPanel({
    required this.step,
    required this.isHindi,
    required this.onAnswer,
    required this.onDone,
    required this.onDismiss,
  });

  @override
  State<_InspectionPanel> createState() => _InspectionPanelState();
}

class _InspectionPanelState extends State<_InspectionPanel>
    with SingleTickerProviderStateMixin {
  late final AnimationController _slideCtrl;
  late final Animation<Offset>   _slideAnim;

  @override
  void initState() {
    super.initState();
    _slideCtrl = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 360));
    _slideAnim = Tween<Offset>(
      begin: const Offset(0, 1), end: Offset.zero,
    ).animate(CurvedAnimation(parent: _slideCtrl, curve: Curves.easeOutCubic));
    _slideCtrl.forward();
  }

  @override
  void dispose() {
    _slideCtrl.dispose();
    super.dispose();
  }

  // Colour tinting: destructive answers get amber/red background
  static Color _optionBg(String id, int idx) {
    // Convention: 'c' or last option is often "significant damage" → red tint
    if (id == 'c') return const Color(0x22FF4B4B);
    if (idx == 0)  return const Color(0x1A34D399); // first = positive → green
    return const Color(0x14FFFFFF);
  }
  static Color _optionBorder(String id, int idx) {
    if (id == 'c') return const Color(0x60FF4B4B);
    if (idx == 0)  return const Color(0x5034D399);
    return const Color(0x30FFFFFF);
  }
  static Color _optionText(String id, int idx) {
    if (id == 'c') return const Color(0xFFFF7070);
    if (idx == 0)  return const Color(0xFF34D399);
    return const Color(0xFFEAEAEA);
  }

  @override
  Widget build(BuildContext context) {
    final step    = widget.step;
    final isHindi = widget.isHindi;
    final isAction = step.isActionStep;

    // Step type icon + label
    final (typeIcon, typeLabel) = switch (step.stepType) {
      StepType.inspection  => (Icons.touch_app_rounded,        isHindi ? 'निरीक्षण'    : 'INSPECTION'),
      StepType.action      => (Icons.build_rounded,             isHindi ? 'कार्य'        : 'ACTION'),
      StepType.observation => (Icons.hearing_rounded,           isHindi ? 'अवलोकन'     : 'OBSERVATION'),
      _                    => (Icons.camera_alt_rounded,        isHindi ? 'दृश्य'        : 'VISUAL'),
    };

    return SlideTransition(
      position: _slideAnim,
      child: Container(
        constraints: BoxConstraints(
          maxHeight: MediaQuery.of(context).size.height * 0.72,
        ),
        decoration: BoxDecoration(
          borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
          gradient: const LinearGradient(
            begin: Alignment.topCenter, end: Alignment.bottomCenter,
            colors: [Color(0xFF1E1E1E), Color(0xFF111111)],
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.70),
              blurRadius: 40, offset: const Offset(0, -10)),
          ],
        ),
        child: SafeArea(
          top: false,
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                // ── Drag handle ─────────────────────────────────────────
                Center(
                  child: Container(
                    width: 36, height: 4,
                    margin: const EdgeInsets.symmetric(vertical: 14),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.16),
                      borderRadius: BorderRadius.circular(2)),
                  ),
                ),

                // ── Step type badge ─────────────────────────────────────
                Row(children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                    decoration: BoxDecoration(
                      color: const Color(0xFF2A2A2A),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: Colors.white.withOpacity(0.12), width: 1)),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(typeIcon, size: 13, color: _C.gold),
                        const SizedBox(width: 5),
                        Text(typeLabel,
                          style: GoogleFonts.inter(
                            fontSize: 11, fontWeight: FontWeight.w700,
                            letterSpacing: 0.8, color: _C.gold)),
                      ],
                    ),
                  ),
                  const Spacer(),
                  // Dismiss — only for non-action steps (action must be confirmed)
                  if (!isAction)
                    GestureDetector(
                      onTap: widget.onDismiss,
                      child: Container(
                        width: 32, height: 32,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.06),
                          shape: BoxShape.circle),
                        child: const Icon(Icons.close_rounded,
                            color: _C.textMuted, size: 16)),
                    ),
                ]),

                const SizedBox(height: 14),

                // ── Step instruction ────────────────────────────────────
                Text(step.getLocalizedText(isHindi),
                  style: GoogleFonts.inter(
                    fontSize: 15, color: _C.textSoft,
                    fontWeight: FontWeight.w500, height: 1.5)),

                // ── Safety warning ──────────────────────────────────────
                if (step.safetyWarning != null) ...[
                  const SizedBox(height: 10),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: _C.warning.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                          color: _C.warning.withOpacity(0.35), width: 1)),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Padding(
                          padding: EdgeInsets.only(top: 2),
                          child: Icon(Icons.warning_amber_rounded,
                              color: _C.warning, size: 15)),
                        const SizedBox(width: 8),
                        Expanded(child: Text(step.safetyWarning!,
                          style: GoogleFonts.inter(
                            fontSize: 12, color: _C.warning,
                            fontWeight: FontWeight.w500, height: 1.4))),
                      ],
                    ),
                  ),
                ],

                const SizedBox(height: 16),

                if (isAction) ...[
                  // ── Action step: single Done button ─────────────────
                  _PanelButton(
                    label:    isHindi ? '✓  कार्य पूर्ण हुआ' : '✓  Action Done',
                    bgColor:  const Color(0x1A34D399),
                    border:   const Color(0x5034D399),
                    text:     const Color(0xFF34D399),
                    onTap:    widget.onDone,
                  ),
                ] else ...[
                  // ── Inspection / observation: question + answer buttons
                  if (step.getLocalizedQuestion(isHindi) != null) ...[
                    Text(step.getLocalizedQuestion(isHindi)!,
                      style: GoogleFonts.inter(
                        fontSize: 16, fontWeight: FontWeight.w700,
                        color: _C.textPrimary, height: 1.3)),
                    const SizedBox(height: 12),
                  ],

                  ...List.generate(step.options.length, (i) {
                    final opt = step.options[i];
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 10),
                      child: _PanelButton(
                        label:   opt.getLocalizedLabel(isHindi),
                        bgColor: _optionBg(opt.id, i),
                        border:  _optionBorder(opt.id, i),
                        text:    _optionText(opt.id, i),
                        onTap:   () => widget.onAnswer(opt),
                      ),
                    );
                  }),
                ],

                const SizedBox(height: 4),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _PanelButton extends StatefulWidget {
  final String label;
  final Color bgColor, border, text;
  final VoidCallback onTap;
  const _PanelButton({
    required this.label, required this.bgColor,
    required this.border, required this.text, required this.onTap});
  @override
  State<_PanelButton> createState() => _PanelButtonState();
}
class _PanelButtonState extends State<_PanelButton> {
  bool _pressed = false;
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown:   (_) => setState(() => _pressed = true),
      onTapUp:     (_) { setState(() => _pressed = false); widget.onTap(); },
      onTapCancel: () => setState(() => _pressed = false),
      child: AnimatedScale(
        scale: _pressed ? 0.97 : 1.0,
        duration: const Duration(milliseconds: 80),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
          decoration: BoxDecoration(
            color: widget.bgColor,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: widget.border, width: 1.2)),
          child: Text(widget.label,
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
              fontSize: 15, fontWeight: FontWeight.w600,
              color: widget.text, height: 1.3)),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Demo steps fallback
// ─────────────────────────────────────────────────────────────────────────
final _demoSteps = List.generate(12, (i) => StepData(
  stepId:      's${i + 1}',
  stepType:    i == 2 ? StepType.inspection
             : i == 6 ? StepType.action
             : i == 9 ? StepType.observation
             : StepType.visual,
  stepTitleEn: 'Step ${i + 1}',
  stepTitleHi: 'चरण ${i + 1}',
  text:      'Inspect component ${i + 1} carefully before proceeding.',
  textEn:    'Inspect component ${i + 1} carefully before proceeding.',
  textHi:    '',
  visualCue: i == 3 ? 'red_cable'
           : i == 4 ? 'fuse_box'
           : i == 5 ? 'battery_terminal'
           : null,
  questionEn: i == 2 ? 'What is the condition of the component?' : null,
  questionHi: i == 2 ? 'घटक की क्या स्थिति है?' : null,
  options: i == 2 ? [
    StepOption(id: 'a', labelEn: 'No damage visible',   labelHi: 'कोई क्षति नहीं', nextStep: 's4'),
    StepOption(id: 'b', labelEn: 'Minor damage',         labelHi: 'मामूली क्षति',   nextStep: 's4'),
    StepOption(id: 'c', labelEn: 'Significant damage',   labelHi: 'गंभीर क्षति',    nextStep: 's_replace'),
    StepOption(id: 'd', labelEn: 'Haven\'t checked yet', labelHi: 'अभी नहीं देखा', nextStep: 's3'),
  ] : [],
));