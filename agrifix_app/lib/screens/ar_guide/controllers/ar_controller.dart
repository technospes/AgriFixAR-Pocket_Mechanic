// lib/screens/ar_guide/controllers/ar_controller.dart
// ignore_for_file: deprecated_member_use
//
// AR pipeline orchestration layer.
//
// Owns: locate loop scheduling, Gemini timing, state transitions,
//       stability counting, detection locking, re-acquire logic,
//       verify-step flow, step navigation, camera lifecycle.
//
// Does NOT own: UI layout, widget trees, CustomPainter drawing.
// Services it delegates to: TrackingService, GuidanceService, TtsService.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';

import '../../../core/providers/diagnosis_provider.dart';
import '../../../core/providers/language_provider.dart';
import '../../../services/api_service.dart';
import '../models/ar_state.dart';
import '../models/bbox.dart';
import '../services/tracking_service.dart';
import '../services/guidance_service.dart';
import '../services/tts_service.dart';

// ── Frame quality gate ─────────────────────────────────────────────────────
// Pure-Dart blur+brightness check. Runs before every Gemini call.
class ARQualityGate {
  static const _kLaplacianMin  = 80.0;
  static const _kBrightnessMin = 30.0;
  static const _kBrightnessMax = 230.0;

  static Future<({bool ok, String message})> check(Uint8List jpeg) async {
    try {
      final codec = await ui.instantiateImageCodec(
          jpeg, targetWidth: 32, targetHeight: 24);
      final frame = await codec.getNextFrame();
      final data  = await frame.image.toByteData(
          format: ui.ImageByteFormat.rawRgba);
      frame.image.dispose();
      if (data == null) return (ok: true, message: '');

      final px = data.buffer.asUint8List();
      const int imgW = 32, imgH = 24, n = imgW * imgH;

      final luma = List<double>.generate(n, (i) {
        final b = i * 4;
        return 0.299 * px[b] + 0.587 * px[b + 1] + 0.114 * px[b + 2];
      });

      var lumaSum = 0.0;
      for (final v in luma) lumaSum += v;
      final brightness = lumaSum / n;

      if (brightness < _kBrightnessMin) {
        return (ok: false, message: 'Too dark — move to better light');
      }
      if (brightness > _kBrightnessMax) {
        return (ok: false,
            message: 'Too bright — avoid direct sunlight on the part');
      }

      var lapSum = 0.0, lapSumSq = 0.0;
      var lapCount = 0;
      for (var y = 1; y < imgH - 1; y++) {
        for (var x = 1; x < imgW - 1; x++) {
          final i = y * imgW + x;
          final v = luma[i - imgW] + luma[i + imgW] +
                    luma[i - 1]    + luma[i + 1] -
                    4.0 * luma[i];
          lapSum   += v;
          lapSumSq += v * v;
          lapCount++;
        }
      }
      final lapMean     = lapSum / lapCount;
      final lapVariance = (lapSumSq / lapCount) - (lapMean * lapMean);

      if (lapVariance < _kLaplacianMin) {
        return (ok: false,
            message: 'Image blurry — hold still and move closer');
      }
      return (ok: true, message: '');
    } catch (_) {
      return (ok: true, message: '');
    }
  }
}

// ── Crop helper ────────────────────────────────────────────────────────────
class ARCropHelper {
  static const _kContextMargin      = 0.40;
  static const _kLargePartThreshold = 0.40;

  static Future<Uint8List?> cropToBbox(Uint8List jpeg, NormBbox bbox) async {
    if (bbox.w * bbox.h > _kLargePartThreshold) return null;
    try {
      final codec = await ui.instantiateImageCodec(jpeg);
      final frame = await codec.getNextFrame();
      final img   = frame.image;
      final imgW  = img.width.toDouble();
      final imgH  = img.height.toDouble();

      final _longestAxis   = math.max(bbox.w, bbox.h);
      final _dynamicMargin = math.min(
          _kContextMargin,
          (0.5 - _longestAxis / 2) * 0.85,
      ).clamp(0.05, _kContextMargin);

      final expW   = bbox.w + 2 * _dynamicMargin;
      final expH   = bbox.h + 2 * _dynamicMargin;
      final left   = ((bbox.cx - expW / 2) * imgW).clamp(0.0, imgW);
      final top    = ((bbox.cy - expH / 2) * imgH).clamp(0.0, imgH);
      final right  = ((bbox.cx + expW / 2) * imgW).clamp(0.0, imgW);
      final bottom = ((bbox.cy + expH / 2) * imgH).clamp(0.0, imgH);
      final cropW  = right - left;
      final cropH  = bottom - top;

      if (cropW < 112 || cropH < 112) { img.dispose(); return null; }

      final recorder = ui.PictureRecorder();
      final canvas   = ui.Canvas(recorder);
      canvas.drawImageRect(
        img,
        Rect.fromLTWH(left, top, cropW, cropH),
        Rect.fromLTWH(0,    0,    cropW, cropH),
        ui.Paint(),
      );
      img.dispose();

      final picture  = recorder.endRecording();
      final cropImg  = await picture.toImage(cropW.round(), cropH.round());
      final byteData = await cropImg.toByteData(format: ui.ImageByteFormat.png);
      cropImg.dispose();

      if (byteData == null) return null;
      return byteData.buffer.asUint8List();
    } catch (e) {
      debugPrint('ARGuide crop failure (using full frame): $e');
      return null;
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════
// ARController
// ══════════════════════════════════════════════════════════════════════════
class ARController {
  // ── Constructor dependencies ──────────────────────────────────────────────
  final BuildContext Function() getContext;
  final void Function(void Function()) setState;
  final bool Function() isMounted;

  // ── Services ──────────────────────────────────────────────────────────────
  final TrackingService tracking  = TrackingService();
  final TtsService      tts       = TtsService();
  late  GuidanceService guidance;

  // ── Scheduling constants (identical to monolith) ─────────────────────────
  static const _kLocateIntervalMs       = 1000;
  static const _kLocateIntervalGuidedMs = 4000;
  static const _kFrameCooldownMs        = 800;
  static const _kReacquireDelayMs       = 500;
  static const _kMaxLocateAttempts      = 8;

  // ── AR state (UI reads these via getters) ─────────────────────────────────
  ARState   arState         = ARState.scanning;
  bool      voiceActive     = false;
  bool      panelExpanded   = false;
  int       attemptCount    = 0;
  String    dangerMessage   = '';
  String    dynamicFeedback = '';
  String    cameraGuidance  = '';
  String    partDescription = '';
  bool      bboxLocked      = false;
  ToastKind toastKind       = ToastKind.analyzing;

  final List<Map<String, dynamic>> attemptResults = [];

  // ── AR pipeline state ─────────────────────────────────────────────────────
  Timer? _locateTimer;
  bool   _locateRunning  = false;
  int    _locateAttempts = 0;
  bool   _partLocked     = false;
  bool   _everDetected   = false;
  DateTime? _lastLocateSent;
  DateTime? _lastCorrectionSent;
  int    _frameId        = 0;
  int    _netFailCount      = 0;  // consecutive network errors
  int    _consecutiveMisses = 0;  // found=False streak during LOCATING
  // Only reset stableFrameCount after _kMissesTolerance consecutive misses.
  // One blurry/occluded frame should not wipe stability progress.
  static const _kMissesTolerance = 2;

  // ── Tracking timer ────────────────────────────────────────────────────────
  Timer?    _trackingTimer;
  bool      _trackingEnabled = false;
  DateTime? _lastBboxTimeTick;
  Timer?    _bboxLockTimer;

  // ── Camera ────────────────────────────────────────────────────────────────
  CameraController? cameraController;
  bool cameraReady      = false;
  bool cameraPermDenied = false;

  // ── Panel state ───────────────────────────────────────────────────────────
  bool inspectionPanelVisible = false;

  // ── Step navigation ───────────────────────────────────────────────────────
  late int currentStep;

  // ── Animation controllers (set by screen after creation) ─────────────────
  late AnimationController bboxFadeCtrl;
  late AnimationController arrowPulseCtrl;
  late AnimationController verifiedCtrl;
  late AnimationController toastCtrl;
  late AnimationController spinnerCtrl;
  late Animation<double>   bboxFadeAnim;
  late Animation<double>   verifiedFade;
  late Animation<double>   toastSlide;
  late Animation<double>   toastFade;
  Timer? toastTimer;

  // ── Orientation ───────────────────────────────────────────────────────────
  Orientation? lastOrientation;

  ARController({
    required this.getContext,
    required this.setState,
    required this.isMounted,
  }) {
    guidance = GuidanceService(
      tts: tts,
      onVisualUpdate: (text) {
        if (isMounted()) setState(() => cameraGuidance = text);
      },
      onVoiceActiveChanged: (active) {
        if (isMounted()) setState(() => voiceActive = active);
      },
    );
  }

  // ══════════════════════════════════════════════════════════════════════════
  // CAMERA
  // ══════════════════════════════════════════════════════════════════════════

  Future<void> initCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      if (isMounted()) setState(() => cameraPermDenied = true);
      return;
    }
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) return;
      cameraController = CameraController(
        cameras.first, ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await cameraController!.initialize();
      try {
        await cameraController!.setFocusMode(FocusMode.auto);
        await cameraController!.setExposureMode(ExposureMode.auto);
      } catch (modeErr) {
        debugPrint('ARGuide: focus/exposure mode not supported — $modeErr');
      }
      if (isMounted()) {
        setState(() => cameraReady = true);
        maybeStartLocateLoop();
      }
    } catch (e) {
      debugPrint('ARGuide: camera init failed — $e');
    }
  }

  Future<void> pauseCamera()  async {
    if (cameraReady) await cameraController?.pausePreview();
  }
  Future<void> resumeCamera() async {
    if (cameraReady) await cameraController?.resumePreview();
  }
  Future<File> captureFrame() async {
    final xFile = await cameraController!.takePicture();
    return File(xFile.path);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // TTS
  // ══════════════════════════════════════════════════════════════════════════

  Future<void> initTts(String langCode) async {
    await tts.init(
      langCode,
      onComplete: () {
        if (isMounted()) setState(() => voiceActive = false);
        guidance.onTtsComplete();
      },
      onCancel: () {
        if (isMounted()) setState(() => voiceActive = false);
        guidance.onTtsComplete();
      },
    );
    // Sync guidance language once — not on every tick.
    // didChangeDependencies calls initTts on start + on locale change,
    // so this fires exactly when it's needed.
    guidance.setLanguage(langCode);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // LOCATE LOOP
  // ══════════════════════════════════════════════════════════════════════════

  void maybeStartLocateLoop() {
    final ctx   = getContext();
    final prov  = ctx.read<DiagnosisProvider>();
    final steps = prov.solution?.steps ?? demoSteps;
    final step  = currentStep < steps.length ? steps[currentStep] : null;
    if (step == null) return;
    if (step.requiresDecisionPanel || step.isActionStep) return;
    if (step.requiredPart.isEmpty && (step.visualCue ?? '').isEmpty) return;
    if (arState == ARState.verified) return;

    _stopLocateLoop();
    _locateAttempts = 0;
    tracking.reset();
    tracking.resetKalman();
    _partLocked         = false;
    _lastCorrectionSent = null;
    guidance.resetForNewSession();
    _consecutiveMisses  = 0;

    if (isMounted()) setState(() => arState = ARState.locating);

    final loopMs = _everDetected ? _kLocateIntervalGuidedMs : _kLocateIntervalMs;
    _locateTimer = Timer.periodic(Duration(milliseconds: loopMs), (_) => _locateTick());
    _locateTick();
  }

  void _stopLocateLoop() {
    _locateTimer?.cancel();
    _locateTimer   = null;
    _locateRunning = false;
    _stopTrackingTimer();
  }

  // ══════════════════════════════════════════════════════════════════════════
  // TRACKING TIMER (30fps Kalman predict)
  // ══════════════════════════════════════════════════════════════════════════

  void _startTrackingTimer() {
    _trackingTimer?.cancel();
    tracking.velCx = 0; tracking.velCy = 0;
    _lastBboxTimeTick = DateTime.now();
    _trackingTimer = Timer.periodic(
      const Duration(milliseconds: 33),
      (_) => _trackingTick(),
    );
  }

  void _stopTrackingTimer() {
    _trackingTimer?.cancel();
    _trackingTimer = null;
  }

  void _trackingTick() {
    if (!isMounted()) return;
    if (!_trackingEnabled) return;
    if (bboxLocked) return;
    if (tracking.smoothBbox == null) return;
    if (arState != ARState.guiding) return;

    final now = DateTime.now();
    final dt  = _lastBboxTimeTick == null
        ? 0.033
        : now.difference(_lastBboxTimeTick!).inMilliseconds / 1000.0;
    _lastBboxTimeTick = now;

    final clampedDt = dt.clamp(0.0, 1.5);
    final predicted = tracking.predictTick(clampedDt);
    if (predicted != null && isMounted()) {
      setState(() {});  // trigger repaint with updated smoothBbox
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // LOCATE TICK — core detection pipeline
  // ══════════════════════════════════════════════════════════════════════════

  Future<void> _locateTick() async {
    if (_locateRunning) return;
    if (!isMounted()) return;
    if (bboxLocked) return;
    if (!cameraReady || cameraController == null) return;
    if (arState == ARState.verified || arState == ARState.analyzing) return;
    final ctx      = getContext();
    final prov     = ctx.read<DiagnosisProvider>();
    final steps    = prov.solution?.steps ?? demoSteps;
    final step     = currentStep < steps.length ? steps[currentStep] : null;
    if (step == null) { _locateRunning = false; return; }
    final langCode = ctx.read<LanguageProvider>().languageCode;
    final machine  = prov.solution?.machineType ?? 'tractor';
    _locateRunning = true;
    _locateAttempts++;
    final now = DateTime.now();

    // ── LOCKED phase: tracking only, no Gemini ───────────────────────────
    if (_partLocked) { _locateRunning = false; return; }

    // ── GUIDING phase: drift-triggered correction ────────────────────────
    if (arState == ARState.guiding &&
        tracking.lastAiBbox != null &&
        tracking.smoothBbox != null) {
      if (!tracking.needsCorrection(_lastCorrectionSent)) {
        debugPrint('ARGuide [SKIP_CORRECTION] tracking accurate');
        _locateRunning = false; return;
      }
    }

    // ── Hard cooldown ─────────────────────────────────────────────────────
    if (_lastLocateSent != null &&
        now.difference(_lastLocateSent!).inMilliseconds < _kFrameCooldownMs) {
      _locateRunning = false; return;
    }

    try {
      final File frame;
      try { frame = await captureFrame(); }
      catch (e) {
        debugPrint('ARGuide captureFrame error: $e');
        _locateRunning = false; return;
      }

      final bytes = await frame.readAsBytes();
      _lastLocateSent = DateTime.now();
      if (arState == ARState.guiding) {
        _lastCorrectionSent = _lastLocateSent;
        debugPrint('ARGuide [CORRECTION] attempt=$_locateAttempts');
      } else {
        debugPrint('ARGuide [DETECT] attempt=$_locateAttempts');
      }

      _frameId++;
      final sentFrameId = _frameId;

      final qResult = await ARQualityGate.check(bytes);
      if (!qResult.ok) {
        if (isMounted()) setState(() => cameraGuidance = qResult.message);
        _locateRunning = false; return;
      }

      final part     = step.requiredPart.isNotEmpty
          ? step.requiredPart : step.visualCue ?? '';
      final area     = step.areaHint;
      if (part.isEmpty) { _locateRunning = false; return; }

      // Pre-detection hint — fires once
      if (tracking.smoothBbox == null) {
        unawaited(guidance.speakPreDetectionHint(
            part, area, isHindi: langCode == 'hi'));
      }

      final roiHint = tracking.smoothBbox != null
          ? '${tracking.smoothBbox!.cx.toStringAsFixed(3)},'
            '${tracking.smoothBbox!.cy.toStringAsFixed(3)},0.300'
          : '';

      Map<String, dynamic> result;
      try {
        result = await ApiService.locatePart(
          imageFile:    frame,
          requiredPart: part,
          areaHint:     area,
          machineType:  machine,
          attemptCount: _locateAttempts,
          language:     langCode,
          frameId:      _frameId,
          searchRoi:    roiHint,
        );
      } catch (e) {
        debugPrint('ARGuide locate_part error: $e');
        _netFailCount++;
        if (_netFailCount >= 2 && isMounted()) {
          setState(() => cameraGuidance = 'Network error — retrying…');
        }
        _locateRunning = false; return;
      }

      if (!isMounted()) { _locateRunning = false; return; }
      if (sentFrameId != _frameId) { _locateRunning = false; return; }
      final respFrameId = result['frame_id'] as int? ?? sentFrameId;
      if (respFrameId != sentFrameId) { _locateRunning = false; return; }
      _netFailCount = 0;

      final found      = result['found'] == true;
      final confidence = (result['confidence'] as num?)?.toDouble() ?? 0.0;
      final rawBbox    = result['bbox'] as List?;
      final rawGuidance = result['camera_guidance'] as String? ?? '';
      final serverGuidance = rawGuidance.contains('Network error') ? '' : rawGuidance;
      final partDesc   = result['part_description'] as String? ?? '';

      if (found && confidence >= TrackingService.kConfThreshold
          && rawBbox != null && rawBbox.length == 4) {
        final cx = (rawBbox[0] as num).toDouble();
        final cy = (rawBbox[1] as num).toDouble();
        final bw = (rawBbox[2] as num).toDouble();
        final bh = (rawBbox[3] as num).toDouble();

        final newSmooth = tracking.update(
            cx: cx, cy: cy, bw: bw, bh: bh, confidence: confidence);
        if (newSmooth == null) {
          debugPrint('ARGuide [BBOX_REJECTED] cx=$cx cy=$cy w=$bw h=$bh '
              'area=${(bw*bh).toStringAsFixed(4)} — failed client sanity');
          _locateRunning = false; return;
        }

        final isHindi = langCode == 'hi';
        final distHint = tracking.distanceHint(bw * bh, isHindi);

        _everDetected = true;
        _consecutiveMisses = 0;  // valid detection — clear miss streak
        setState(() {
          partDescription = partDesc;
          cameraGuidance  = distHint.isNotEmpty ? distHint : '';
        });

        // ── DIAGNOSTIC: print after every valid detection ─────────────
        debugPrint('═══ ARGuide DETECT: STATE=${arState.name} '
            'STABLE=${tracking.stableFrameCount}/${TrackingService.kStableFramesNeeded} '
            'conf=${confidence.toStringAsFixed(2)} '
            'bbox=(${newSmooth.cx.toStringAsFixed(3)},${newSmooth.cy.toStringAsFixed(3)}) '
            'ARROW=${arState == ARState.guiding ? "VISIBLE" : "waiting for stable"}');
        debugPrint('ARGuide [KALMAN] est=(${newSmooth.cx},${newSmooth.cy})');

        // Directional guidance (confidence-gated)
        if (confidence >= GuidanceService.kGuidanceConfThreshold) {
          unawaited(guidance.speakDirectionalHint(
              newSmooth, isHindi: langCode == 'hi'));
        }

        // ── LOCKED transition ──────────────────────────────────────────────
        if (tracking.stableFrameCount >= TrackingService.kLockFramesNeeded &&
            confidence >= TrackingService.kLockConfThreshold &&
            !_partLocked) {
          _partLocked = true;
          debugPrint('ARGuide [LOCKED] stable=${tracking.stableFrameCount}');
          guidance.bypassDedup();
          unawaited(guidance.speakGuidance(
            'Locked — hold steady and tap Analyze Part',
            'भाग लॉक — स्थिर रखें और "भाग विश्लेषण करें" दबाएं',
          ));
        }

        // ── GUIDING transition ─────────────────────────────────────────────
        if (tracking.stableFrameCount >= TrackingService.kStableFramesNeeded &&
            (arState == ARState.locating || arState == ARState.unclear)) {
          bboxFadeCtrl.forward(from: 0);
          setState(() {
            arState        = ARState.guiding;
            cameraGuidance = '';
          });
          HapticFeedback.mediumImpact();
          _stopLocateLoop();
          _trackingEnabled = true;
          _consecutiveMisses = 0;  // fresh tracking session
          _startTrackingTimer();
          _bboxLockTimer?.cancel();
          _bboxLockTimer = Timer(const Duration(milliseconds: 1500), () {});
          debugPrint('══════════════════════════════════════════════');
          debugPrint('ARGuide [GUIDING] ✅ ARROW SHOULD NOW BE VISIBLE');
          debugPrint('  stable=${tracking.stableFrameCount} locked=$_partLocked');
          debugPrint('  bboxFadeCtrl started — arrow fading in over 350ms');
          debugPrint('══════════════════════════════════════════════');

          // Part Found announcement
          final label = partDesc.isNotEmpty
              ? partDesc
              : (step.requiredPart).replaceAll('_', ' ');
          guidance.bypassDedup();
          unawaited(guidance.speakGuidance(
            label.isNotEmpty
                ? 'Found it — tap Analyze Part to verify'
                : 'Part found — tap Analyze Part to verify',
            label.isNotEmpty
                ? 'मिल गया — जांच के लिए "भाग विश्लेषण करें" दबाएं'
                : 'भाग मिल गया — "भाग विश्लेषण करें" दबाएं',
          ));
        }

        // ── CORRECTION timer restart ───────────────────────────────────────
        if (arState == ARState.guiding && _locateTimer == null && !_partLocked) {
          _locateTimer = Timer.periodic(
              const Duration(milliseconds: _kLocateIntervalGuidedMs),
              (_) => _locateTick());
        }

      } else {
        // ── Detection lost / re-acquisition ───────────────────────────────
        final wasLocked = _partLocked;
        _consecutiveMisses++;
        final bool shouldResetStability =
            arState == ARState.guiding ||        // always reset if was guiding
            _consecutiveMisses >= _kMissesTolerance; // or after 2 misses
        debugPrint('ARGuide [FOUND=FALSE] STATE=${arState.name} '
            'miss_streak=$_consecutiveMisses '
            'stable=${tracking.stableFrameCount} '
            '→ ${shouldResetStability ? "RESET" : "PRESERVED"}');
        _trackingEnabled = false;
        _partLocked      = false;
        tracking.resetVelocity();
        if (shouldResetStability) tracking.stableFrameCount = 0;
        guidance.resetOnDetectionLost();

        if (wasLocked) {
          debugPrint('ARGuide [REACQUIRE] re-acquiring…');
        }
        setState(() {
          cameraGuidance = serverGuidance.isNotEmpty ? serverGuidance : '';
          if (arState == ARState.guiding) {
            arState          = ARState.locating;
            tracking.smoothBbox = null;
            bboxFadeCtrl.reverse();
            _stopLocateLoop();
            _lastCorrectionSent = null;
            Future.delayed(
              const Duration(milliseconds: _kReacquireDelayMs),
              () {
                if (isMounted() && arState == ARState.locating) {
                  final raMs = _everDetected
                      ? _kLocateIntervalGuidedMs : _kLocateIntervalMs;
                  _locateTimer = Timer.periodic(
                      Duration(milliseconds: raMs), (_) => _locateTick());
                  _locateTick();
                }
              },
            );
          }
        });

        if (_locateAttempts >= _kMaxLocateAttempts) {
          _stopLocateLoop();
          if (isMounted()) {
            // ✅ Cleaned up unused variables and removed unnecessary ? and !
            final part2 = (step.requiredPart.isNotEmpty 
                ? step.requiredPart : (step.visualCue ?? ''))
                .replaceAll('_', ' ');
            final area2 = step.areaHint.replaceAll('_', ' ');
            final hindi = langCode == 'hi';

            setState(() {
              dynamicFeedback = hindi
                  ? 'स्वचालित रूप से नहीं मिला।\n'
                    '👉 देखें: $part2${area2.isNotEmpty ? " → $area2" : ""}\n'
                    'कैमरा वहाँ ले जाएं और "भाग विश्लेषण करें" दबाएं।'
                  : 'Could not locate automatically.\n'
                    '👉 Look for: $part2${area2.isNotEmpty ? " at $area2" : ""}\n'
                    'Point camera there and tap Analyze Part.';
              arState = ARState.scanning;
            });
          }
        }
      }
    } finally {
      _locateRunning = false;
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // TOAST
  // ══════════════════════════════════════════════════════════════════════════

  Duration _dismissDuration(ToastKind k) {
    switch (k) {
      case ToastKind.analyzing:  return const Duration(seconds: 30);
      case ToastKind.sent:       return const Duration(seconds: 5);
      case ToastKind.analyzed:   return const Duration(seconds: 5);
      case ToastKind.resultOk:   return const Duration(seconds: 6);
      case ToastKind.resultWarn: return const Duration(seconds: 8);
      case ToastKind.error:      return const Duration(seconds: 8);
    }
  }

  Future<void> showToast(ToastKind kind) async {
    if (!isMounted()) return;
    if (toastCtrl.isAnimating || toastCtrl.isCompleted) {
      await toastCtrl.reverse();
    }
    if (!isMounted()) return;
    setState(() => toastKind = kind);
    toastCtrl.forward(from: 0);
    _scheduleToastDismiss(_dismissDuration(kind));
  }

  void _scheduleToastDismiss(Duration delay) {
    toastTimer?.cancel();
    toastTimer = Timer(delay, () async {
      if (!isMounted()) return;
      await toastCtrl.reverse();
      if (isMounted() && arState == ARState.unclear) {
        transitionTo(ARState.scanning);
      }
    });
  }

  // ══════════════════════════════════════════════════════════════════════════
  // STATE TRANSITIONS
  // ══════════════════════════════════════════════════════════════════════════

  void transitionTo(ARState next) {
    if (!isMounted()) return;
    setState(() => arState = next);
    switch (next) {
      case ARState.scanning:
      case ARState.locating:
        toastTimer?.cancel();
        toastCtrl.reverse();
        break;
      case ARState.guiding:
        break;
      case ARState.analyzing:
        break;
      case ARState.unclear:
        break;
      case ARState.verified:
        _stopLocateLoop();
        _partLocked         = false;
        _lastCorrectionSent = null;
        toastTimer?.cancel();
        toastCtrl.reverse();
        verifiedCtrl.forward(from: 0);
        break;
      case ARState.danger:
        _stopLocateLoop();
        toastTimer?.cancel();
        toastCtrl.reset();
        break;
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // CAPTURE / VERIFY
  // ══════════════════════════════════════════════════════════════════════════

  Future<void> onCapture() async {
    if (arState == ARState.analyzing) return;
    if (!cameraReady || cameraController == null) return;
    final ctx   = getContext();
    final prov  = ctx.read<DiagnosisProvider>();
    final steps = prov.solution?.steps ?? demoSteps;
    final step  = currentStep < steps.length ? steps[currentStep] : null;
    final machine  = prov.solution?.machineType ?? 'tractor';
    final ctx2     = getContext();
    final isHindi  = ctx2.read<LanguageProvider>().languageCode == 'hi';
    final problem  = prov.solution?.getLocalizedProblem(isHindi) ?? '';
    if (step != null && (step.requiresDecisionPanel || step.isActionStep)) {
      setState(() => inspectionPanelVisible = true);
      return;
    }

    _stopLocateLoop();
    _partLocked         = false;
    _lastCorrectionSent = null;
    setState(() => bboxLocked = true);
    HapticFeedback.mediumImpact();
    attemptCount++;
    await pauseCamera();
    transitionTo(ARState.analyzing);
    await showToast(ToastKind.analyzing);

    final File imageFile;
    try {
      imageFile = await captureFrame();
    } catch (_) {
      await resumeCamera();
      transitionTo(ARState.locating);
      maybeStartLocateLoop();
      return;
    }

    await showToast(ToastKind.sent);

    final verifyBytes = await imageFile.readAsBytes();
    final vq = await ARQualityGate.check(verifyBytes);
    if (!vq.ok) {
      await resumeCamera();
      setState(() {
        bboxLocked      = false;
        dynamicFeedback = vq.message;
      });
      await showToast(ToastKind.resultWarn);
      transitionTo(ARState.unclear);
      Future.delayed(const Duration(seconds: 2), maybeStartLocateLoop);
      return;
    }

    final cropBytes = tracking.smoothBbox != null
        ? await ARCropHelper.cropToBbox(verifyBytes, tracking.smoothBbox!)
        : null;

    final stepText = step?.textEn.isNotEmpty == true
        ? step!.textEn : step?.text ?? '';
    final reqPart  = step?.requiredPart.isNotEmpty == true
        ? step!.requiredPart : (step?.visualCue ?? '');
    final areaHint = step?.areaHint ?? '';

    Map<String, dynamic> result;
    try {
      result = await ApiService.verifyStep(
        imageFile:      imageFile,
        imageCropBytes: cropBytes,
        stepText:       stepText,
        machineType:    machine,
        problemContext: problem,
        attemptCount:   attemptCount,
        requiredPart:   reqPart,
        areaHint:       areaHint,
        previousSteps:  jsonEncode(attemptResults),
      );
    } on Exception catch (e) {
      debugPrint('ARGuide verifyStep error: $e');
      await showToast(ToastKind.error);
      await resumeCamera();
      setState(() => bboxLocked = false);
      transitionTo(ARState.locating);
      Future.delayed(const Duration(seconds: 2), maybeStartLocateLoop);
      return;
    }

    await showToast(ToastKind.analyzed);

    final isDangerous = result['danger'] == true ||
        result['status'] == 'danger' ||
        (result['severity'] as String? ?? '').toLowerCase() == 'critical';

    if (isDangerous) {
      if (isMounted()) {
        setState(() {
          dangerMessage = result['danger_message'] as String? ??
              'STOP — Critical safety hazard detected!\n\n'
              'The machine appears to be running or there is an immediate risk.\n'
              'Do NOT proceed until the machine is fully off and safe.';
        });
      }
      transitionTo(ARState.danger);
      HapticFeedback.vibrate();
      return;
    }

    final verified = result['verified'] == true ||
        result['status'] == 'verified' || result['correct'] == true;

    if (verified) {
      await showToast(ToastKind.resultOk);
      transitionTo(ARState.verified);
      HapticFeedback.heavyImpact();
    } else {
      attemptResults.add({
        'attempt_count': result['attempt_count'] ?? attemptCount,
        'status':        result['status']        ?? 'unclear',
        'detected_part': result['detected_part'] ?? '',
        'feedback':      result['feedback']      ?? '',
      });
      if (isMounted()) {
        // ✅ Removed ctx3 and hindi3 entirely!
        setState(() {
          final raw = isHindi
              ? (result['feedback_hi'] ?? result['feedback'])
              : result['feedback'];
          dynamicFeedback = raw ??
              result['ai_observation'] ??
              'Image unclear or wrong part captured — see hint below for guidance.';
        });
      }
      await showToast(ToastKind.resultWarn);
      await resumeCamera();
      setState(() => bboxLocked = false);
      transitionTo(ARState.unclear);
      Future.delayed(const Duration(seconds: 2), maybeStartLocateLoop);
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // STEP NAVIGATION
  // ══════════════════════════════════════════════════════════════════════════

  void nextPart(List<StepData> steps) {
    if (currentStep + 1 >= steps.length) return;
    HapticFeedback.lightImpact();
    _stopLocateLoop();
    resumeCamera();
    setState(() {
      currentStep             = currentStep + 1;
      arState                 = ARState.scanning;
      attemptCount            = 0;
      panelExpanded           = false;
      dynamicFeedback         = '';
      cameraGuidance          = '';
      partDescription         = '';
      bboxLocked              = false;
      _partLocked             = false;
      _lastCorrectionSent     = null;
      _netFailCount           = 0;
      _everDetected           = false;
      _trackingEnabled        = false;
      inspectionPanelVisible  = false;
    });
    tracking.reset();
    guidance.resetOnStepChange();
    attemptResults.clear();
    verifiedCtrl.reset();
    bboxFadeCtrl.reset();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!isMounted()) return;
      final ctx   = getContext();
      final prov  = ctx.read<DiagnosisProvider>();
      final stps  = prov.solution?.steps ?? demoSteps;
      if (currentStep < stps.length) {
        final s = stps[currentStep];
        if (s.requiresDecisionPanel || s.isActionStep) {
          setState(() => inspectionPanelVisible = true);
        } else {
          maybeStartLocateLoop();
        }
      }
    });
  }

  void onInspectionAnswer(StepOption option, List<StepData> steps) {
    HapticFeedback.mediumImpact();
    attemptResults.add({
      'attempt_count': 1,
      'status':        'answered',
      'detected_part': option.id,
      'feedback':      option.labelEn,
    });
    setState(() => inspectionPanelVisible = false);

    if (option.nextStep.isNotEmpty) {
      final ctx      = getContext();
      final prov     = ctx.read<DiagnosisProvider>();
      final targetIdx = prov.solution?.indexOfStepId(option.nextStep) ?? -1;
      if (targetIdx >= 0 && targetIdx < steps.length) {
        Future.delayed(const Duration(milliseconds: 300), () {
          if (!isMounted()) return;
          _stopLocateLoop();
          setState(() {
            currentStep             = targetIdx;
            arState                 = ARState.scanning;
            attemptCount            = 0;
            panelExpanded           = false;
            dynamicFeedback         = '';
            cameraGuidance          = '';
            bboxLocked              = false;
            _partLocked             = false;
            _lastCorrectionSent     = null;
            _trackingEnabled        = false;
            _everDetected           = false;
            inspectionPanelVisible  = false;
          });
          tracking.reset();
          guidance.resetOnStepChange();
          attemptResults.clear();
          verifiedCtrl.reset();
          bboxFadeCtrl.reset();
          final nextS = steps[targetIdx];
          if (nextS.requiresDecisionPanel || nextS.isActionStep) {
            setState(() => inspectionPanelVisible = true);
          } else {
            maybeStartLocateLoop();
          }
        });
        return;
      }
    }

    Future.delayed(const Duration(milliseconds: 300), () {
      if (!isMounted()) return;
      transitionTo(ARState.verified);
      HapticFeedback.heavyImpact();
    });
  }

  void onActionDone() {
    HapticFeedback.heavyImpact();
    setState(() => inspectionPanelVisible = false);
    transitionTo(ARState.verified);
  }

  void maybeShowInspectionPanel(StepData? step) {
    if (step == null) return;
    if ((step.requiresDecisionPanel || step.isActionStep) &&
        !inspectionPanelVisible &&
        arState == ARState.scanning) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (isMounted()) setState(() => inspectionPanelVisible = true);
      });
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // ORIENTATION CHANGE
  // ══════════════════════════════════════════════════════════════════════════

  void onOrientationChange() {
    setState(() {
      tracking.smoothBbox       = null;
      tracking.prevBbox         = null;
      tracking.stableFrameCount = 0;
      bboxLocked                = false;
      if (arState == ARState.guiding) arState = ARState.locating;
    });
    _stopTrackingTimer();
    _partLocked         = false;
    _lastCorrectionSent = null;
    tracking.resetKalman();
    tracking.resetVelocity();
    bboxFadeCtrl.reverse();
    if (arState != ARState.verified) maybeStartLocateLoop();
  }

  // ══════════════════════════════════════════════════════════════════════════
  // DISPOSE
  // ══════════════════════════════════════════════════════════════════════════

  void dispose() {
    _locateTimer?.cancel();
    _bboxLockTimer?.cancel();
    _trackingTimer?.cancel();
    toastTimer?.cancel();
    tts.stop();
    cameraController?.dispose();
  }
}

// ── Demo steps fallback ────────────────────────────────────────────────────
final demoSteps = List.generate(12, (i) => StepData(
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
  requiredPart: i == 3 ? 'red_cable'
              : i == 4 ? 'fuse_box'
              : i == 5 ? 'battery_terminal'
              : '',
  areaHint:     i >= 3 && i <= 5 ? 'electrical_panel' : '',
  questionEn: i == 2 ? 'What is the condition of the component?' : null,
  questionHi: i == 2 ? 'घटक की क्या स्थिति है?' : null,
  options: i == 2 ? [
    StepOption(id: 'a', labelEn: 'No damage visible',   labelHi: 'कोई क्षति नहीं', nextStep: 's4'),
    StepOption(id: 'b', labelEn: 'Minor damage',         labelHi: 'मामूली क्षति',   nextStep: 's4'),
    StepOption(id: 'c', labelEn: 'Significant damage',   labelHi: 'गंभीर क्षति',    nextStep: 's_replace'),
    StepOption(id: 'd', labelEn: "Haven't checked yet",  labelHi: 'अभी नहीं देखा', nextStep: 's3'),
  ] : [],
));