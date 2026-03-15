// ignore_for_file: deprecated_member_use
import 'dart:async';
// ignore: depend_on_referenced_packages
import 'package:meta/meta.dart' show visibleForTesting;
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
// AR Guide Screen — v4.3 (Production AR — smart Gemini scheduling)
//
// v4.2 — Production accuracy hardening:
//   • _ARQualityGate runs before verifyStep (motion-blur on capture blocked)
//   • _frameId + server echo → two-layer stale response discard
//   • _bboxLockTimer: 1.5s lock window after bbox stabilises
//   • Orientation change → instant bbox + stability reset
//   • Close-up/too-far UX hints from bboxArea
//   • frame_id wired through Flutter → API → backend → response
// v4.1 — All 12 production fixes applied (quality gate, conf threshold,
//         stability, bbox sanity, EMA jump reset, orientation, lock mode…)
// v4.0 — Initial AR guidance system
// v3.1 — Corner bracket artifact fix
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
  static const arrowGreen  = Color(0xFF22C55E);
  static const arrowBlue   = Color(0xFF38BDF8);
}

enum _ARState  { scanning, locating, guiding, analyzing, unclear, verified, danger }
enum _ToastKind { analyzing, sent, analyzed, resultOk, resultWarn, error }

const double _kPanelHeight = 240.0;
const double _kCaptureSize = 72.0;
const double _kBoxSize     = 300.0;


// ═══════════════════════════════════════════════════════════════════════════
// _ARCropHelper — crops the captured JPEG to the bbox region before verify
//
// Workflow:
//   camera frame → AR bbox (already known) → crop to part region
//   → send cropped image to verify_step
//
// Why this matters:
//   1. Higher effective resolution — Gemini sees the part at full pixel
//      density rather than downscaled from 1440×1080.
//   2. Fewer distractors — background machinery, other pipes, soil removed.
//   3. Smaller upload — ~40 KB crop vs ~350 KB full frame (~8× smaller).
//   4. Faster Gemini reasoning — fewer tokens, less background to parse.
//
// Context margin (40%): bbox is expanded so Gemini sees the part AND its
// immediate surroundings — critical for "check where this joint connects".
//
// Fallback: if bbox covers > 40% of image (large part, close-up), the full
// frame is used — no crop benefit for parts that already fill the view.
// On any decode/canvas error the full frame is used silently.
//
// Encoding: PNG via dart:ui — zero extra packages, Gemini accepts natively.
// A 300×300 crop PNG ≈ 60 KB (still 6× smaller than full JPEG).
// ═══════════════════════════════════════════════════════════════════════════
class _ARCropHelper {
  static const _kContextMargin      = 0.40;  // expand bbox by 40% for context
  static const _kLargePartThreshold = 0.40;  // skip crop if bbox area > 40%

  /// Returns the cropped PNG bytes, or null to fall back to the full frame.
  static Future<Uint8List?> cropToBbox(
    Uint8List jpeg,
    _NormBbox bbox,
  ) async {
    // Large part — already fills most of the frame, no crop benefit
    if (bbox.w * bbox.h > _kLargePartThreshold) return null;

    try {
      // Decode full JPEG to get actual pixel dimensions
      final codec = await ui.instantiateImageCodec(jpeg);
      final frame = await codec.getNextFrame();
      final img   = frame.image;
      final imgW  = img.width.toDouble();
      final imgH  = img.height.toDouble();

      // FIX 3: Dynamic margin — scales down for larger bboxes so the expanded
      // region never pushes outside the image frame on either axis.
      // Formula: min(0.40, (0.5 - longestAxis/2) * 0.85)
      // Guarantees expW and expH stay ≤ ~0.95, preserving part centring.
      final _longestAxis  = math.max(bbox.w, bbox.h);
      final _dynamicMargin = math.min(
          _kContextMargin,                          // never exceed 40%
          (0.5 - _longestAxis / 2) * 0.85,          // shrinks for larger parts
      ).clamp(0.05, _kContextMargin);               // never below 5%
      final expW   = bbox.w + 2 * _dynamicMargin;  // absolute expansion, not relative
      final expH   = bbox.h + 2 * _dynamicMargin;
      final left   = ((bbox.cx - expW / 2) * imgW).clamp(0.0, imgW);
      final top    = ((bbox.cy - expH / 2) * imgH).clamp(0.0, imgH);
      final right  = ((bbox.cx + expW / 2) * imgW).clamp(0.0, imgW);
      final bottom = ((bbox.cy + expH / 2) * imgH).clamp(0.0, imgH);
      final cropW  = right - left;
      final cropH  = bottom - top;

      // FIX 4: minimum 112×112 px — 32px is decodable but too small for
      // Gemini Vision to reliably identify a specific machine part.
      // At 112px the part fills ~7 of Gemini's 16×16 processing tiles.
      if (cropW < 112 || cropH < 112) {
        img.dispose();
        return null;  // fall back to full frame — better than a tiny crop
      }

      // Draw the crop region onto a new canvas
      final recorder = ui.PictureRecorder();
      final canvas   = ui.Canvas(recorder);
      canvas.drawImageRect(
        img,
        Rect.fromLTWH(left, top, cropW, cropH),   // source
        Rect.fromLTWH(0,    0,    cropW, cropH),   // dest
        ui.Paint(),
      );
      img.dispose();

      final picture  = recorder.endRecording();
      final cropImg  = await picture.toImage(cropW.round(), cropH.round());
      final byteData = await cropImg.toByteData(
          format: ui.ImageByteFormat.png);
      cropImg.dispose();

      if (byteData == null) return null;
      return byteData.buffer.asUint8List();
    } catch (e) {
      debugPrint('ARGuide crop failure (using full frame): $e');
      return null;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _ARQualityGate — FIX 1
//
// Pure-Dart frame quality check. Decodes the captured JPEG to 32×24 RGBA
// using dart:ui (~2 ms decode), then computes:
//
//   • Laplacian variance (blur metric)
//     Kernel: [0,1,0 / 1,−4,1 / 0,1,0] applied to Y-luma channel.
//     At 32×24 the kernel pass takes ~1 ms on low-end Android.
//     Variance < 80 → reject as blurry.
//
//   • Luminance mean (brightness)
//     Y = 0.299R + 0.587G + 0.114B  (BT.601)
//     mean < 30  → too dark (shadowed engine bay, night)
//     mean > 230 → overexposed (direct sun on chrome/metal)
//
// Why file-size is wrong:
//   - Dark images → small JPEG (low entropy) but not blurry
//   - Noisy images → larger JPEG but ARE blurry to Gemini
//   - Threshold would need constant re-calibration per device/camera
//
// This gate runs BEFORE the backend call — zero token cost on bad frames.
// ═══════════════════════════════════════════════════════════════════════════
class _ARQualityGate {
  static const _kLaplacianMin   = 80.0;   // variance below this → blurry
  static const _kBrightnessMin  = 30.0;   // luma mean below this → too dark
  static const _kBrightnessMax  = 230.0;  // luma mean above this → overexposed

  /// Returns (ok: true) if the frame passes all quality checks.
  /// Returns (ok: false, message: "<user hint>") otherwise.
  static Future<({bool ok, String message})> check(Uint8List jpeg) async {
    try {
      // Decode to tiny 32×24 RGBA — resolution-agnostic for Laplacian variance
      final codec = await ui.instantiateImageCodec(
        jpeg, targetWidth: 32, targetHeight: 24);
      final frame = await codec.getNextFrame();
      final data  = await frame.image.toByteData(
          format: ui.ImageByteFormat.rawRgba);
      frame.image.dispose();
      if (data == null) return (ok: true, message: '');

      final px = data.buffer.asUint8List();
      const int imgW = 32, imgH = 24, n = imgW * imgH;

      // Compute Y-luma for each pixel (BT.601)
      final luma = List<double>.generate(n, (i) {
        final b = i * 4;
        return 0.299 * px[b] + 0.587 * px[b + 1] + 0.114 * px[b + 2];
      });

      // ── Brightness ──────────────────────────────────────────────────────
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

      // ── Laplacian variance ───────────────────────────────────────────────
      // Apply 3×3 Laplacian to interior pixels only (skip 1-px border)
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
      // On any decode error, pass the frame through — backend reject flags
      // will catch remaining quality issues (image_too_blurry flag).
      return (ok: true, message: '');
    }
  }
}

// ── Bbox model (normalised 0.0–1.0) ───────────────────────────────────────
class _NormBbox {
  final double cx, cy, w, h;
  const _NormBbox(this.cx, this.cy, this.w, this.h);

  _NormBbox lerp(_NormBbox other, double t) => _NormBbox(
    cx + (other.cx - cx) * t,
    cy + (other.cy - cy) * t,
    w  + (other.w  - w ) * t,
    h  + (other.h  - h ) * t,
  );

  double distanceTo(_NormBbox other) {
    final dx = cx - other.cx, dy = cy - other.cy;
    final dw = w  - other.w,  dh = h  - other.h;
    return math.sqrt(dx*dx + dy*dy + dw*dw + dh*dh);
  }

  Rect toScreenRect(double previewW, double previewH) => Rect.fromCenter(
    center: Offset(cx * previewW, cy * previewH),
    width:  w * previewW,
    height: h * previewH,
  );
}

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

  // ── AR Guidance state ─────────────────────────────────────────────────────
  // ── Production AR architecture constants ────────────────────────────────
  // PHASE 1 — DETECTION  : Gemini called until 2 stable frames → guiding
  // PHASE 2 — TRACKING   : 30fps local velocity extrapolation, no Gemini
  // PHASE 3 — LOCKED     : ≥4 stable, conf≥0.85 → Gemini fully stopped
  // PHASE 4 — CORRECTION : Gemini every 2 s while guiding (not locked)
  // PHASE 5 — RE-ACQUIRE : bbox lost → stop tracking → call Gemini again

  // Detection thresholds
  static const _kConfThreshold       = 0.82;  // min conf to accept a bbox
  static const _kLockConfThreshold   = 0.85;  // conf needed to enter LOCKED phase
  static const _kStabilityThreshold  = 0.04;  // max bbox delta for stable frame
  static const _kStableFramesNeeded  = 2;     // frames to enter guiding
  static const _kLockFramesNeeded    = 4;     // frames to enter LOCKED (stop Gemini)
  // Scheduling constants
  // Locate intervals — 1 s for first detection (find part fast),
  // 4 s after first success (Gemini budget preserved during guidance).
  static const _kLocateIntervalMs      = 1000;  // ticker base
  static const _kLocateIntervalGuidedMs= 4000;  // interval once part found once
  static const _kFrameCooldownMs       = 800;   // hard cooldown between Gemini calls
  static const _kReacquireDelayMs      = 500;   // pause before re-acquire after loss
  static const _kMaxLocateAttempts   = 8;     // failures before showing manual hint

  // Tracking constants
  static const _kMaxVelocity         = 0.5;   // max normalised/s velocity
  static const _kJumpThreshold       = 0.25;  // bbox jump that resets Kalman+stability
  // 0.25 (not 0.20): during LOCATING, the phone naturally moves >20% while
  // panning to find the part. 0.20 was resetting stableCount on every tick.
  static const _kVelocityResetDist   = 0.15;  // jump distance that zeros velocity

  // AR Phase enum — drives Gemini scheduling
  // locating  : Gemini every tick (part not yet found)
  // guiding   : Gemini on drift > _kDriftThreshold OR every _kMaxCorrectionMs (arrow visible)
  // locked    : Gemini stopped (stable ≥4, conf ≥0.85)
  // reacquire : Gemini immediately (part lost from guiding/locked)

  // State
  Timer?       _locateTimer;
  bool         _locateRunning  = false;
  int          _locateAttempts = 0;
  _NormBbox?   _smoothBbox;
  _NormBbox?   _prevBbox;
  int          _stableFrameCount = 0;
  bool         _partLocked       = false;  // LOCKED phase — no Gemini calls
  DateTime?    _lastLocateSent;
  DateTime?    _lastCorrectionSent;        // last Gemini call while guiding
  late final AnimationController _arrowPulseCtrl;
  late final AnimationController _bboxFadeCtrl;
  late final Animation<double>   _bboxFadeAnim;
  String _cameraGuidance  = '';
  String _partDescription = '';
  bool   _bboxLocked      = false; // frozen when user taps Verify
  int    _frameId         = 0;
  // ── Two-stage tracking: velocity prediction between AI calls ────────────
  // Stage 1 (AI): /locate_part every ~1s → ground-truth bbox
  // Stage 2 (local): 30fps Timer predicts next position using last velocity
  // This makes the arrow track at 30fps without extra API calls.
  // ── Kalman filter state (per axis, cx and cy independently) ────────────────
  // x_est: current best position estimate
  // P:     estimate error covariance (uncertainty)
  // Q:     process noise — how much the part can move between ticks (0.001)
  // R:     measurement noise — Gemini bbox uncertainty (~1.5% of image = 0.015)
  // At init P is high (1.0) so first Gemini reading is trusted fully.
  // After a correction P drops, so prediction is trusted until drift grows again.
  static const _kKalmanQ = 0.001;  // process noise  — part moves slowly
  static const _kKalmanR = 0.015;  // measurement noise — Gemini ~1.5% bbox error
  double _kfCxEst = 0.0, _kfCyEst = 0.0;  // Kalman estimate
  double _kfPCx   = 1.0, _kfPCy   = 1.0;  // error covariance (1.0 = uninitialised)
  bool   _kfInitialised = false;            // false until first Gemini measurement

  // ── Velocity + tracking infrastructure ──────────────────────────────────────
  double _velCx = 0.0, _velCy = 0.0;  // bbox centre velocity (normalised/s)
  DateTime? _lastBboxTime;              // time of last AI bbox update
  Timer?    _trackingTimer;             // 30fps prediction loop
  bool      _trackingEnabled = false;
  int       _netFailCount    = 0;      // consecutive network errors

  // ── Drift-triggered correction state ────────────────────────────────────────
  // _lastAiBbox: the _smoothBbox value at the time of the last Gemini correction.
  // Drift = distance from _lastAiBbox to current _smoothBbox.
  // If drift < 0.08 AND msSinceCorrection < 4000: skip Gemini (tracking is accurate).
  // If drift ≥ 0.08 OR msSinceCorrection ≥ 4000: call Gemini (correction needed).
  static const _kDriftThreshold  = 0.08;  // normalised image fraction
  static const _kMaxCorrectionMs = 4000;  // force correction every 4s regardless
  _NormBbox? _lastAiBbox;               // bbox at last AI correction

  // ── Predictive Guidance Engine ─────────────────────────────────────────────
  // Provides spoken + visual instructions BEFORE the bbox is stable.
  // Phase 1 (pre-detection): rule-based hints from area_hint metadata.
  // Phase 2 (bbox visible): directional hints from bbox centre position.
  // TTS cooldown: 5 s between spoken instructions.
  static const _kTtsCooldownMs         = 5000;  // min ms between TTS instructions
  static const _kGuidanceConfThreshold = 0.87;  // min conf to speak directional hint
  // Below this threshold the bbox is too uncertain to reliably guide the farmer.
  static const _kBboxLeftThresh    = 0.35;  // cx < 0.35 → 'move camera right'
  static const _kBboxRightThresh   = 0.65;  // cx > 0.65 → 'move camera left'
  static const _kBboxTopThresh     = 0.35;  // cy < 0.35 → 'move camera down'
  static const _kBboxBottomThresh  = 0.65;  // cy > 0.65 → 'move camera up'
  DateTime? _lastGuidanceSpoke;     // TTS cooldown timestamp
  String    _lastGuidanceText = ''; // deduplicate: don't repeat same phrase
  bool      _everDetected = false;  // true after first valid detection
  bool      _preDetectionHintFired = false; // speak pre-detect hint once per session
  Timer? _bboxLockTimer;           // BBox lock window — auto-relock after guiding
  Orientation? _lastOrientation;  // track rotation → reset bbox on change

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
      // REQ 11: Use medium (720p) not veryHigh.
      // medium = 1280×720 — sufficient for Gemini Vision part detection.
      // veryHigh on budget phones can be 4K → massive JPEG → slow uploads.
      // Clutch pedal at 720p is still clearly distinguishable.
      _cameraController = CameraController(
        cameras.first, ResolutionPreset.medium,
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

      if (mounted) {
        setState(() => _cameraReady = true);
        // Start locate loop for visual steps
        _maybeStartLocateLoop();
      }
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


  // ═══════════════════════════════════════════════════════════════════════════
  // AR LOCATE LOOP
  // ═══════════════════════════════════════════════════════════════════════════

  void _maybeStartLocateLoop() {
    final prov  = context.read<DiagnosisProvider>();
    final steps = prov.solution?.steps ?? _demoSteps;
    final step  = _currentStep < steps.length ? steps[_currentStep] : null;
    if (step == null) return;
    if (step.requiresDecisionPanel || step.isActionStep) return;
    if (step.requiredPart.isEmpty && (step.visualCue ?? '').isEmpty) return;
    if (_arState == _ARState.verified) return;
    _stopLocateLoop();
    _locateAttempts      = 0;
    _stableFrameCount    = 0;
    _smoothBbox          = null;
    _prevBbox            = null;
    _partLocked          = false;
    _lastCorrectionSent  = null;
    _lastAiBbox          = null;
    _kfInitialised       = false;
    _kfPCx = 1.0; _kfPCy = 1.0;  // reset to high uncertainty
    _lastGuidanceText        = '';    // allow guidance to speak on loop restart
    _preDetectionHintFired   = false; // fire pre-detection hint again on re-acquire
    if (mounted) setState(() => _arState = _ARState.locating);
    // Use 4s interval once part has been detected at least once —
    // saves Gemini budget while farmer moves to verify position.
    final _loopMs = _everDetected ? _kLocateIntervalGuidedMs : _kLocateIntervalMs;
    _locateTimer = Timer.periodic(
        Duration(milliseconds: _loopMs), (_) => _locateTick());
    _locateTick();  // fire immediately
  }

  void _stopLocateLoop() {
    _locateTimer?.cancel();
    _locateTimer   = null;
    _locateRunning = false;
    _stopTrackingTimer();
  }

  // ── Stage 2 tracking: 30fps velocity-prediction loop ─────────────────────
  // Between AI ground-truth calls, predicts next bbox position using the
  // velocity computed from the last two AI detections. Gives the arrow a
  // smooth 30fps update rate with zero backend cost.
  //
  // Why not optical flow? startImageStream() conflicts with takePicture() on
  // many Android OEM camera implementations — both require the same camera
  // session. Velocity extrapolation is simpler, reliable, and fast enough
  // for the typical stationary-phone + slowly-moving-pointer use case.
  void _startTrackingTimer() {
    _trackingTimer?.cancel();
    _velCx = 0; _velCy = 0;  // reset on each new guiding session
    _lastBboxTime = DateTime.now();
    _trackingTimer = Timer.periodic(
      const Duration(milliseconds: 33),  // ~30fps
      (_) => _trackingTick(),
    );
  }

  void _stopTrackingTimer() {
    _trackingTimer?.cancel();
    _trackingTimer = null;
  }

  // ══════════════════════════════════════════════════════════════════════
  // Predictive Guidance Engine
  // ══════════════════════════════════════════════════════════════════════
  //
  // Phase 1 — Pre-detection (part not yet found):
  //   Uses area_hint + requiredPart metadata → rule-based spoken instruction.
  //   "Look near the floor under the steering wheel."
  //
  // Phase 2 — Bbox visible but not centred:
  //   Compares bbox centre to screen centre thresholds.
  //   "Move camera slightly right", "Move camera up", etc.
  //
  // Phase 3 — Centred:
  //   "Good — hold camera steady." (once)
  //
  // TTS cooldown: _kTtsCooldownMs (5 s) — never speaks more frequently.
  // Deduplication: same phrase is never repeated back-to-back.
  // ══════════════════════════════════════════════════════════════════════

  /// Speak a camera guidance instruction if cooldown allows.
  Future<void> _speakGuidance(String en, String hi) async {
    if (!mounted) return;
    final isHindi = context.read<LanguageProvider>().languageCode == 'hi';
    final text    = isHindi ? hi : en;
    if (text.isEmpty) return;

    // Deduplicate — skip if same phrase as last spoken
    if (text == _lastGuidanceText) return;

    // TTS cooldown — never speak more often than _kTtsCooldownMs
    final now = DateTime.now();
    if (_lastGuidanceSpoke != null &&
        now.difference(_lastGuidanceSpoke!).inMilliseconds < _kTtsCooldownMs) {
      return;
    }

    _lastGuidanceSpoke = now;
    _lastGuidanceText  = text;

    // Update visual chip immediately (no cooldown for visual)
    if (mounted) setState(() => _cameraGuidance = text);

    // Speak if TTS not already active
    if (!_voiceActive) {
      setState(() => _voiceActive = true);
      await _tts.speak(text);
    }
  }

  // ── Phase 1: rule-based pre-detection hints ────────────────────────────
  // Called each tick when _smoothBbox is null (part not yet found).
  // Translates area_hint metadata into plain directional instructions.
  // No Gemini call needed — pure metadata lookup.
  Future<void> _speakPreDetectionHint(String requiredPart, String areaHint) async {
    final hint = _preDetectionHint(requiredPart, areaHint);
    if (hint == null) return;
    await _speakGuidance(hint.en, hint.hi);
  }

  static ({String en, String hi})? _preDetectionHint(
      String part, String area) {
    // Maps area_hint → farmer-friendly direction instruction.
    // Keep these SHORT — TTS speaks fast and farmers need to act quickly.
    const Map<String, ({String en, String hi})> areaMap = {
      'transmission_area': (
        en: 'Look near the floor under the steering wheel',
        hi: 'स्टीयरिंग व्हील के नीचे फर्श के पास देखें'
      ),
      'engine_compartment': (
        en: 'Open the hood and point camera at the engine',
        hi: 'बोनट खोलें और इंजन पर कैमरा करें'
      ),
      'undercarriage': (
        en: 'Point camera underneath the machine',
        hi: 'मशीन के नीचे कैमरा करें'
      ),
      'fuel_system': (
        en: 'Look near the fuel tank and fuel lines',
        hi: 'ईंधन टंकी और ईंधन पाइपों के पास देखें'
      ),
      'electrical_panel': (
        en: 'Look for the fuse box or wiring panel',
        hi: 'फ्यूज बॉक्स या वायरिंग पैनल ढूंढें'
      ),
      'hydraulic_system': (
        en: 'Look near the hydraulic pump and pipes',
        hi: 'हाइड्रोलिक पंप और पाइपों के पास देखें'
      ),
      'cooling_system': (
        en: 'Point camera at the radiator at the front',
        hi: 'सामने रेडिएटर पर कैमरा करें'
      ),
      'air_intake': (
        en: 'Look for the air filter box near the engine top',
        hi: 'इंजन के ऊपर एयर फिल्टर बॉक्स ढूंढें'
      ),
      'exhaust_system': (
        en: 'Look for the exhaust pipe at the side or back',
        hi: 'साइड या पीछे एग्जॉस्ट पाइप ढूंढें'
      ),
      'pto_area': (
        en: 'Look at the rear of the tractor near the PTO shaft',
        hi: 'ट्रैक्टर के पीछे पीटीओ शाफ्ट के पास देखें'
      ),
      'front_axle': (
        en: 'Point camera at the front wheels and axle',
        hi: 'अगले पहियों और एक्सल पर कैमरा करें'
      ),
      'rear_axle': (
        en: 'Point camera at the rear wheels and axle',
        hi: 'पिछले पहियों और एक्सल पर कैमरा करें'
      ),
    };

    // Check part-specific overrides first
    const Map<String, ({String en, String hi})> partMap = {
      'clutch_pedal': (
        en: 'Look at the pedals on the left side of the floor',
        hi: 'फर्श के बाईं ओर पेडल देखें'
      ),
      'clutch_linkage': (
        en: 'Look underneath near where the pedal connects',
        hi: 'पेडल के नीचे लिंकेज ढूंढें'
      ),
      'fuel_filter': (
        en: 'Look for a small cylindrical filter near the fuel line',
        hi: 'ईंधन लाइन के पास छोटा बेलनाकार फ़िल्टर ढूंढें'
      ),
      'air_filter': (
        en: 'Look for a round filter box on top of the engine',
        hi: 'इंजन के ऊपर गोल फ़िल्टर बॉक्स ढूंढें'
      ),
      'battery': (
        en: 'Look for the battery near the engine, usually has thick cables',
        hi: 'इंजन के पास बैटरी ढूंढें — मोटी तारें होंगी'
      ),
      'radiator': (
        en: 'Point camera at the front grille of the machine',
        hi: 'मशीन के सामने ग्रिल पर कैमरा करें'
      ),
    };

    final partHint = partMap[part.toLowerCase().replaceAll(' ', '_')];
    if (partHint != null) return partHint;
    return areaMap[area.toLowerCase().replaceAll(' ', '_')];
  }

  // ── Phase 2: directional hints from bbox centre position ─────────────
  // Called when bbox is visible but not centred.
  // Thresholds: cx < 0.35 → right, cx > 0.65 → left,
  //             cy < 0.35 → down,  cy > 0.65 → up.
  // Note: cx < 0.35 means part is LEFT of centre, so camera needs to go RIGHT.
  Future<void> _speakDirectionalHint(_NormBbox bbox) async {
    final cx = bbox.cx;
    final cy = bbox.cy;
    final bool centredX = cx >= _kBboxLeftThresh && cx <= _kBboxRightThresh;
    final bool centredY = cy >= _kBboxTopThresh  && cy <= _kBboxBottomThresh;

    if (centredX && centredY) {
      await _speakGuidance(
        'Good — hold camera steady',
        'अच्छा — कैमरा स्थिर रखें',
      );
      return;
    }

    // ── Combined instruction: both axes can be off simultaneously ────────
    // "Move camera down and right" is more natural than two separate calls.
    String hEn = '', hHi = '', vEn = '', vHi = '';

    if (!centredX) {
      if (cx < _kBboxLeftThresh) {
        hEn = 'right'; hHi = 'दाईं ओर';
      } else {
        hEn = 'left';  hHi = 'बाईं ओर';
      }
    }
    if (!centredY) {
      if (cy < _kBboxTopThresh) {
        vEn = 'down'; vHi = 'नीचे';
      } else {
        vEn = 'up';   vHi = 'ऊपर';
      }
    }

    final String en, hi;
    if (hEn.isNotEmpty && vEn.isNotEmpty) {
      en = 'Move camera $vEn and $hEn';
      hi = 'कैमरा $vHi और $hHi ले जाएं';
    } else if (hEn.isNotEmpty) {
      en = 'Move camera slightly $hEn';
      hi = 'कैमरा थोड़ा $hHi ले जाएं';
    } else {
      en = 'Move camera $vEn';
      hi = 'कैमरा $vHi करें';
    }
    await _speakGuidance(en, hi);
  }



  // Kalman PREDICT — only active in GUIDING state.
  // During LOCATING: tracking timer is not even running (only starts on
  // guiding transition). The _arState guard is a belt-and-suspenders check.
  void _trackingTick() {
    if (!mounted) return;
    if (!_trackingEnabled) return;
    if (_bboxLocked) return;
    if (_smoothBbox == null) return;
    if (_arState != _ARState.guiding) return;  // prediction only in guiding

    final now = DateTime.now();
    final dt  = _lastBboxTime == null
        ? 0.033
        : now.difference(_lastBboxTime!).inMilliseconds / 1000.0;

    // Clamp dt — if AI call was slow, don't extrapolate more than 1.5s
    // to avoid overshooting drastically.
    final clampedDt = dt.clamp(0.0, 1.5);

    // Only advance if velocity is meaningful (> 0.001 normalised/s)
    final speed = math.sqrt(_velCx * _velCx + _velCy * _velCy);
    if (speed < 0.001) return;  // stationary — no Kalman update needed

    // ── Kalman PREDICT step ───────────────────────────────────────────
    // x_pred = x_est + vel*dt;  P_pred = P_est + Q  (uncertainty grows)
    if (!_kfInitialised) return;  // wait for first Gemini measurement
    final predCx = (_kfCxEst + _velCx * clampedDt).clamp(0.01, 0.99);
    final predCy = (_kfCyEst + _velCy * clampedDt).clamp(0.01, 0.99);
    _kfPCx += _kKalmanQ;  // covariance grows: we trust Gemini more next time
    _kfPCy += _kKalmanQ;
    _kfCxEst = predCx;
    _kfCyEst = predCy;
    final kalmanPred = _NormBbox(predCx, predCy, _smoothBbox!.w, _smoothBbox!.h);
    if (mounted) setState(() => _smoothBbox = kalmanPred);
  }

  Future<void> _locateTick() async {
    if (_locateRunning) return;
    if (!mounted) return;
    if (_bboxLocked) return;                      // FIX 11: locked → no backend queries
    if (!_cameraReady || _cameraController == null) return;
    if (_arState == _ARState.verified || _arState == _ARState.analyzing) return;
    _locateRunning = true;
    _locateAttempts++;
    final _now = DateTime.now();

    // ── Gemini call scheduling ────────────────────────────────────────────
    // LOCKED phase: part stable ≥4 frames, conf ≥0.85 → no Gemini at all.
    // Tracking timer handles arrow at 30fps with zero API cost.
    if (_partLocked) {
      _locateRunning = false; return;
    }

    // GUIDING phase: drift-triggered correction.
    // Gemini is only called if tracking has drifted beyond _kDriftThreshold (0.08)
    // OR the max correction interval (_kMaxCorrectionMs = 4 s) has elapsed.
    // Between corrections the Kalman prediction handles arrow movement.
    // This reduces guiding Gemini calls by ~40-60% vs fixed-interval scheduling.
    if (_arState == _ARState.guiding && _lastAiBbox != null && _smoothBbox != null) {
      final msSinceCorrection = _lastCorrectionSent == null
          ? 9999
          : _now.difference(_lastCorrectionSent!).inMilliseconds;
      final drift = _smoothBbox!.distanceTo(_lastAiBbox!);
      final driftOk  = drift < _kDriftThreshold;        // tracking still accurate
      final timerOk  = msSinceCorrection < _kMaxCorrectionMs; // not overdue
      if (driftOk && timerOk) {
        // No correction needed — Kalman prediction is tracking well
        debugPrint('ARGuide [SKIP_CORRECTION] drift=${drift.toStringAsFixed(3)} < $_kDriftThreshold, '
            'ms=$msSinceCorrection < $_kMaxCorrectionMs');
        _locateRunning = false; return;
      }
      if (!driftOk) debugPrint('ARGuide [DRIFT] ${drift.toStringAsFixed(3)} >= $_kDriftThreshold → correction');
    }

    // Hard cooldown — never send faster than _kFrameCooldownMs regardless of phase
    if (_lastLocateSent != null &&
        _now.difference(_lastLocateSent!).inMilliseconds < _kFrameCooldownMs) {
      _locateRunning = false; return;
    }
    try {
      final File frame;
      try { frame = await _captureFrame(); }
      catch (e) { debugPrint('ARGuide captureFrame error: $e'); _locateRunning = false; return; }

      // FIX 1: real blur + brightness gate (Laplacian variance on 32×24 px decode)
      final bytes   = await frame.readAsBytes();
      _lastLocateSent = DateTime.now();  // mark cooldown timestamp
      if (_arState == _ARState.guiding) {
        _lastCorrectionSent = _lastLocateSent;
        debugPrint('ARGuide [CORRECTION] guiding correction call — attempt=$_locateAttempts');
      } else {
        debugPrint('ARGuide [DETECT] locate call — attempt=$_locateAttempts');
      }
      _frameId++;                        // new frame — stale responses will be dropped
      final _sentFrameId = _frameId;
      final qResult = await _ARQualityGate.check(bytes);
      if (!qResult.ok) {
        if (mounted) setState(() => _cameraGuidance = qResult.message);
        _locateRunning = false;
        return;
      }

      final prov  = context.read<DiagnosisProvider>();
      final steps = prov.solution?.steps ?? _demoSteps;
      final step  = _currentStep < steps.length ? steps[_currentStep] : null;
      if (step == null) { _locateRunning = false; return; }

      final langCode = context.read<LanguageProvider>().languageCode;
      final machine  = prov.solution?.machineType ?? 'tractor';
      final part     = step.requiredPart.isNotEmpty ? step.requiredPart : step.visualCue ?? '';
      final area     = step.areaHint;
      if (part.isEmpty) { _locateRunning = false; return; }

      // Pre-detection guidance: speak once per session while part not found.
      // Fires on the very first tick so the farmer gets instant direction.
      // Does NOT repeat — one instruction is enough; constant reminders are annoying.
      if (_smoothBbox == null && !_preDetectionHintFired) {
        _preDetectionHintFired = true;
        unawaited(_speakPreDetectionHint(part, area));
      }

      Map<String, dynamic> result;
      try {
        // ROI hint: tell Gemini where the part was last seen
        // Provides a ±0.30 search window around the last known bbox centre.
        // Reduces false positives and speeds up Gemini reasoning.
        final _roiHint = _smoothBbox != null
            ? '${_smoothBbox!.cx.toStringAsFixed(3)},'
              '${_smoothBbox!.cy.toStringAsFixed(3)},0.300'
            : '';
        result = await ApiService.locatePart(
          imageFile:    frame,
          requiredPart: part,
          areaHint:     area,
          machineType:  machine,
          attemptCount: _locateAttempts,
          language:     langCode,
          frameId:      _frameId,
          searchRoi:    _roiHint,
        );
      } catch (e) {
        debugPrint('ARGuide locate_part error: $e');
        _netFailCount++;
        // Only show 'network error' chip after 2 consecutive failures
        // — single failures from dual-IP switching are transient and noisy.
        if (_netFailCount >= 2 && mounted) {
          setState(() => _cameraGuidance = 'Network error — retrying…');
        }
        _locateRunning = false; return;
      }

      if (!mounted) { _locateRunning = false; return; }
      // Discard stale response — a newer frame was already sent (local counter)
      if (_sentFrameId != _frameId) { _locateRunning = false; return; }
      // Also check server-echoed frame_id (catches delayed network responses)
      final respFrameId = result['frame_id'] as int? ?? _sentFrameId;
      if (respFrameId != _sentFrameId) { _locateRunning = false; return; }
      // Reset network fail counter on any successful HTTP response
      _netFailCount = 0;

      final found      = result['found'] == true;
      final confidence = (result['confidence'] as num?)?.toDouble() ?? 0.0;
      final rawBbox    = result['bbox'] as List?;
      // Filter out the catch-block fallback message — it's not real camera guidance
      final rawGuidance = result['camera_guidance'] as String? ?? '';
      final guidance   = rawGuidance.contains('Network error') ? '' : rawGuidance;
      final partDesc   = result['part_description'] as String? ?? '';

      if (found && confidence >= _kConfThreshold && rawBbox != null && rawBbox.length == 4) {
        // ── FIX 5: Strict coordinate validation ────────────────────────────
        final cx = (rawBbox[0] as num).toDouble();
        final cy = (rawBbox[1] as num).toDouble();
        final bw = (rawBbox[2] as num).toDouble();
        final bh = (rawBbox[3] as num).toDouble();
        // All coords must be strictly inside (0, 1)
        if (cx <= 0 || cx >= 1 || cy <= 0 || cy >= 1 ||
            bw <= 0 || bw >= 1 || bh <= 0 || bh >= 1) {
          _locateRunning = false; return; // invalid coords — Gemini hallucination
        }
        // Bbox must not spill outside the image frame
        if (cx - bw / 2 < 0 || cx + bw / 2 > 1 ||
            cy - bh / 2 < 0 || cy + bh / 2 > 1) {
          _locateRunning = false; return;
        }
        // ── FIX 9: Bbox size sanity ────────────────────────────────────────
        // Reject pinpoint boxes (w/h < 3%) and full-image boxes (w/h > 60%).
        // Also require total bbox area in [2%, 60%] of image.
        final bboxArea = bw * bh;
        // Area threshold: 0.009 (0.9%) not 0.02.
        // A clutch_pedal at w=0.10, h=0.18 = 1.8% was being rejected.
        // Individual w/h >= 0.03 checks already prevent degenerate boxes;
        // area only needs to block a 0.03×0.03 trivial box (0.09%).
        if (bw < 0.03 || bh < 0.03 || bw > 0.60 || bh > 0.60 ||
            bboxArea < 0.009 || bboxArea > 0.60) {
          _locateRunning = false; return;
        }

        // ── Kalman UPDATE step ────────────────────────────────────────────
        // First measurement: initialise filter with Gemini's value directly.
        // Subsequent measurements: blend prediction with measurement via gain K.
        //
        //   K = P_pred / (P_pred + R)        → Kalman gain (0=trust prediction, 1=trust Gemini)
        //   x_est = x_pred + K*(z - x_pred)  → updated estimate
        //   P_est = (1-K) * P_pred            → updated covariance
        //
        // After many tracking ticks P_pred is large → K is high → trust Gemini.
        // Right after a correction P_est is small → K is low → trust prediction.

        final double newCx, newCy;
        if (!_kfInitialised) {
          // First detection: accept directly, reset covariance to low value
          _kfCxEst = cx; _kfCyEst = cy;
          _kfPCx   = _kKalmanR; _kfPCy = _kKalmanR;
          _kfInitialised = true;
          newCx = cx; newCy = cy;
        } else {
          // Subsequent detection: Kalman update
          final kGainCx = _kfPCx / (_kfPCx + _kKalmanR);
          final kGainCy = _kfPCy / (_kfPCy + _kKalmanR);
          newCx    = _kfCxEst + kGainCx * (cx - _kfCxEst);
          newCy    = _kfCyEst + kGainCy * (cy - _kfCyEst);
          _kfCxEst = newCx;
          _kfCyEst = newCy;
          _kfPCx   = (1 - kGainCx) * _kfPCx;
          _kfPCy   = (1 - kGainCy) * _kfPCy;
          debugPrint('ARGuide [KALMAN] K=(${kGainCx.toStringAsFixed(3)},${kGainCy.toStringAsFixed(3)}) '
              'est=($newCx,$newCy) meas=($cx,$cy)');
        }

        // Jump detection using Kalman estimate (not raw measurement)
        final jumpDist = _smoothBbox == null ? 1.0 : math.sqrt(
          math.pow(newCx - _smoothBbox!.cx, 2) +
          math.pow(newCy - _smoothBbox!.cy, 2),
        );
        final didJump = jumpDist > _kJumpThreshold;
        if (didJump) {
          // Large jump: reset Kalman to new measurement, re-initialise
          _kfCxEst = cx; _kfCyEst = cy;
          _kfPCx   = _kKalmanR; _kfPCy = _kKalmanR;
        }
        final newSmooth = _NormBbox(newCx.clamp(0.01, 0.99), newCy.clamp(0.01, 0.99), bw, bh);
        if (didJump) _stableFrameCount = 0;

        // First detection has no previous bbox to compare — treat as stable.
        // Using 1.0 before required 3 calls (not 2) to show the arrow.
        final delta = _prevBbox == null ? 0.0 : newSmooth.distanceTo(_prevBbox!);
        if (delta < _kStabilityThreshold) { _stableFrameCount++; }
        else { _stableFrameCount = 0; }

        // ── UX: distance hints based on bbox area ─────────────────────────
        final _isHindi = context.read<LanguageProvider>().languageCode == 'hi';
        String _distHint = '';
        if (bboxArea > 0.55) {
          _distHint = _isHindi
              ? 'थोड़ा पीछे हटें — भाग बहुत करीब है'
              : 'Move phone back slightly — part fills too much of frame';
        } else if (bboxArea < 0.009) {
          // Only show 'move closer' for truly tiny bboxes —
          // not for small-but-valid parts like clutch_pedal (area ~1.8%)
          _distHint = _isHindi
              ? 'मशीन के और करीब जाएं'
              : 'Move closer to the machine';
        }
        // Compute velocity from consecutive AI detections.
        // Reset to zero whenever the bbox jumps > 0.15 (perspective change /
        // camera swing) — a jump that large means the computed velocity would
        // be enormous and _trackingTick would overshoot the new position.
        if (_prevBbox != null && _lastBboxTime != null) {
          final _jumpDist2 = math.sqrt(
            math.pow(newSmooth.cx - _prevBbox!.cx, 2) +
            math.pow(newSmooth.cy - _prevBbox!.cy, 2),
          );
          if (_jumpDist2 > _kVelocityResetDist) {
            // Large perspective jump — zero velocity so tracking holds still
            _velCx = 0.0;
            _velCy = 0.0;
          } else {
            final dtSec = DateTime.now().difference(_lastBboxTime!).inMilliseconds / 1000.0;
            if (dtSec > 0.05) {  // only compute if > 50ms since last update
              _velCx = (newSmooth.cx - _prevBbox!.cx) / dtSec;
              _velCy = (newSmooth.cy - _prevBbox!.cy) / dtSec;
              // Clamp max speed — if > _kMaxVelocity norm/s it's a jump not motion
              _velCx = _velCx.clamp(-_kMaxVelocity, _kMaxVelocity);
              _velCy = _velCy.clamp(-_kMaxVelocity, _kMaxVelocity);
            }
          }
        }
        _lastBboxTime = DateTime.now();

        _everDetected = true;  // enables 4s interval from next cycle
        setState(() {
          _smoothBbox      = newSmooth;
          _prevBbox        = newSmooth;
          _lastAiBbox      = newSmooth;   // drift baseline — measured from here
          _partDescription = partDesc;
          _cameraGuidance  = _distHint.isNotEmpty ? _distHint : '';
        });
        // Directional guidance — only speak when confidence is high enough.
        // A low-confidence bbox could point the farmer the wrong direction.
        if (confidence >= _kGuidanceConfThreshold) {
          unawaited(_speakDirectionalHint(newSmooth));
        }
        // ── Phase transitions ───────────────────────────────────────────
        // LOCKED: stable ≥4 frames, conf ≥0.85 → stop ALL Gemini calls.
        if (_stableFrameCount >= _kLockFramesNeeded &&
            confidence >= _kLockConfThreshold &&
            !_partLocked) {
          _partLocked = true;
          debugPrint('ARGuide [LOCKED] part=$partDesc stable=$_stableFrameCount conf=$confidence');
          // ── Part Locked announcement ──────────────────────────────
          _lastGuidanceText = '';  // bypass dedup — unique lock event
          unawaited(_speakGuidance(
            'Locked — hold steady and tap Analyze Part',
            'भाग लॉक — स्थिर रखें और "भाग विश्लेषण करें" दबाएं',
          ));
        }

        // GUIDING: stable ≥2 frames → show arrow, start tracking
        if (_stableFrameCount >= _kStableFramesNeeded &&
            (_arState == _ARState.locating || _arState == _ARState.unclear)) {
          _bboxFadeCtrl.forward(from: 0);
          setState(() {
            _arState        = _ARState.guiding;
            // Show clear tap instruction — the 'Scanning...' pill disappears
            // here so the user needs to know what to do next.
            _cameraGuidance = '';
          });
          HapticFeedback.mediumImpact();
          _stopLocateLoop();
          _trackingEnabled = true;
          _startTrackingTimer();
          _bboxLockTimer?.cancel();
          _bboxLockTimer = Timer(const Duration(milliseconds: 1500), () {});
          debugPrint('ARGuide [GUIDING] stable=$_stableFrameCount locked=$_partLocked');
          // ── Part Found announcement ───────────────────────────────
          // Speak immediately, bypassing dedup so it always fires on
          // first guiding entry regardless of what was spoken before.
          final _label = partDesc.isNotEmpty
              ? partDesc
              : (step?.requiredPart ?? '').replaceAll('_', ' ');
          _lastGuidanceText = '';  // bypass dedup — this is a unique event
          unawaited(_speakGuidance(
            _label.isNotEmpty
                ? 'Found it — tap Analyze Part to verify'
                : 'Part found — tap Analyze Part to verify',
            _label.isNotEmpty
                ? 'मिल गया — जांच के लिए "भाग विश्लेषण करें" दबाएं'
                : 'भाग मिल गया — "भाग विश्लेषण करें" दबाएं',
          ));
        }

        // CORRECTION: still guiding → restart locate timer for periodic corrections
        if (_arState == _ARState.guiding && _locateTimer == null && !_partLocked) {
          _locateTimer = Timer.periodic(
              const Duration(milliseconds: _kLocateIntervalGuidedMs), (_) => _locateTick());
        }
      } else {
        // ── Detection lost / re-acquisition ────────────────────────────
        final wasLocked = _partLocked;
        _trackingEnabled    = false;
        _partLocked         = false;   // unlock — re-acquire with Gemini
        _velCx = 0.0; _velCy = 0.0;
        _stableFrameCount   = 0;
        _lastGuidanceText        = '';   // allow guidance to speak immediately on re-acquire
        _preDetectionHintFired   = false;
        if (wasLocked) {
          debugPrint('ARGuide [REACQUIRE] part was locked — starting re-acquisition');
        }
        setState(() {
          _cameraGuidance = guidance.isNotEmpty ? guidance : '';
          if (_arState == _ARState.guiding) {
            _arState    = _ARState.locating;
            _smoothBbox = null;
            _bboxFadeCtrl.reverse();
            _stopLocateLoop();
            _lastCorrectionSent = null;  // reset so next detection is immediate
            // Brief pause before re-acquisition so camera can settle
            Future.delayed(
              const Duration(milliseconds: _kReacquireDelayMs),
              () {
                if (mounted && _arState == _ARState.locating) {
                  final _raMs = _everDetected ? _kLocateIntervalGuidedMs : _kLocateIntervalMs;
                  _locateTimer = Timer.periodic(
                      Duration(milliseconds: _raMs),
                      (_) => _locateTick());
                  _locateTick();
                }
              },
            );
          }
        });
        if (_locateAttempts >= _kMaxLocateAttempts) {
          _stopLocateLoop();
          if (mounted) {
            // FIX 7: surface part name + area so farmer knows exactly where to look
            final _prov2  = context.read<DiagnosisProvider>();
            final _steps2 = _prov2.solution?.steps ?? _demoSteps;
            final _step2  = _currentStep < _steps2.length ? _steps2[_currentStep] : null;
            final _part2  = (_step2?.requiredPart.isNotEmpty == true
                ? _step2!.requiredPart : (_step2?.visualCue ?? ''))
                .replaceAll('_', ' ');
            final _area2  = (_step2?.areaHint ?? '').replaceAll('_', ' ');
            final _hindi  = context.read<LanguageProvider>().languageCode == 'hi';
            setState(() {
              _dynamicFeedback = _hindi
                  ? 'स्वचालित रूप से नहीं मिला।\n'
                    '👉 देखें: $_part2${_area2.isNotEmpty ? " → $_area2" : ""}\n'
                    'कैमरा वहाँ ले जाएं और "भाग विश्लेषण करें" दबाएं।'
                  : 'Could not locate automatically.\n'
                    '👉 Look for: $_part2${_area2.isNotEmpty ? " at $_area2" : ""}\n'
                    'Point camera there and tap Analyze Part.';
              _arState = _ARState.scanning;
            });
          }
        }
      }
    } finally {
      _locateRunning = false;
    }
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

    // AR-specific controllers
    _arrowPulseCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 900))
      ..repeat(reverse: true);
    _bboxFadeCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 350));
    _bboxFadeAnim = CurvedAnimation(
        parent: _bboxFadeCtrl, curve: Curves.easeOut);

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
    _locateTimer?.cancel();
    _bboxLockTimer?.cancel();
    _trackingTimer?.cancel();
    _tts.stop();
    _toastTimer?.cancel();
    _toastCtrl.dispose();
    _verifiedCtrl.dispose();
    _spinnerCtrl.dispose();
    _arrowPulseCtrl.dispose();
    _bboxFadeCtrl.dispose();
    _cameraController?.dispose();
    super.dispose();
  }

  Duration _dismissDuration(_ToastKind k) {
    switch (k) {
      case _ToastKind.analyzing:  return const Duration(seconds: 30);
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
      case _ARState.locating:
        _toastTimer?.cancel();
        _toastCtrl.reverse();
        break;
      case _ARState.guiding:
        break;
      case _ARState.analyzing:
        break;
      case _ARState.unclear:
        break;
      case _ARState.verified:
        _stopLocateLoop();
        _partLocked = false;   // clear lock — next step starts fresh
        _lastCorrectionSent = null;
        _toastTimer?.cancel();
        _toastCtrl.reverse();
        _verifiedCtrl.forward(from: 0);
        break;
      case _ARState.danger:
        _stopLocateLoop();
        _toastTimer?.cancel();
        _toastCtrl.reset();
        break;
    }
  }

  Future<void> _onCapture() async {
    // Allow capture in scanning, guiding (part found → verify), or unclear (retry)
    if (_arState == _ARState.analyzing) return;
    if (!_cameraReady || _cameraController == null) return;

    // Non-visual steps use the inspection / action panel, not the camera
    final prov  = context.read<DiagnosisProvider>();
    final steps = prov.solution?.steps ?? _demoSteps;
    final step  = _currentStep < steps.length ? steps[_currentStep] : null;
    if (step != null && (step.requiresDecisionPanel || step.isActionStep)) {
      setState(() => _inspectionPanelVisible = true);
      return;
    }

    _stopLocateLoop();
    _partLocked = false;                // unlock so loop restarts cleanly after verify
    _lastCorrectionSent = null;
    setState(() => _bboxLocked = true); // freeze arrow during verify
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
      _transitionTo(_ARState.locating);
      _maybeStartLocateLoop();
      return;
    }

    await _showToast(_ToastKind.sent);

    // ── Quality gate + bbox crop before verifyStep ────────────────────────
    // 1. Quality check on the full frame (blur + brightness)
    // 2. If bbox is available (guiding state), crop to part region
    //    → Gemini sees only the part area: higher resolution, fewer
    //      distractors, ~8× smaller upload, faster/more accurate verify.
    final _verifyBytes = await imageFile.readAsBytes();
    final _vq = await _ARQualityGate.check(_verifyBytes);
    if (!_vq.ok) {
      await _resumeCamera();
      setState(() {
        _bboxLocked      = false;
        _dynamicFeedback = _vq.message;
      });
      await _showToast(_ToastKind.resultWarn);
      _transitionTo(_ARState.unclear);
      Future.delayed(const Duration(seconds: 2), _maybeStartLocateLoop);
      return;
    }
    // Attempt bbox crop when the AR system already knows where the part is.
    // Falls back to null (full frame) if bbox not available or part too large.
    final _cropBytes = (_smoothBbox != null && !_bboxLocked)
        ? await _ARCropHelper.cropToBbox(_verifyBytes, _smoothBbox!)
        : null;

    final stepText = step?.textEn.isNotEmpty == true
        ? step!.textEn : step?.text ?? '';
    final machine  = prov.solution?.machineType ?? 'tractor';
    final isHindi = context.read<LanguageProvider>().languageCode == 'hi';
    final problem  = prov.solution?.getLocalizedProblem(isHindi) ?? '';

    // Extract real step metadata — never fall back to backend defaults
    final _requiredPart = step?.requiredPart.isNotEmpty == true
        ? step!.requiredPart
        : (step?.visualCue ?? '');
    final _areaHint = step?.areaHint ?? '';

    Map<String, dynamic> result;
    try {
      result = await ApiService.verifyStep(
        imageFile:        imageFile,          // fallback if crop unavailable
        imageCropBytes:   _cropBytes,         // PNG crop — null → use full frame
        stepText:         stepText,
        machineType:      machine,
        problemContext:   problem,
        attemptCount:     _attemptCount,
        requiredPart:     _requiredPart,
        areaHint:         _areaHint,
        previousSteps:    jsonEncode(_attemptResults),
      );
    } on Exception catch (e) {
      debugPrint('ARGuide verifyStep error: $e');
      await _showToast(_ToastKind.error);
      await _resumeCamera();
      setState(() => _bboxLocked = false);
      _transitionTo(_ARState.locating);
      // Restart locate loop so arrow reappears after a network error
      Future.delayed(const Duration(seconds: 2), _maybeStartLocateLoop);
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
      setState(() => _bboxLocked = false); // FIX 11: unlock so loop can restart
      _transitionTo(_ARState.unclear);
      // Restart locate loop after unclear so arrow can reappear
      Future.delayed(const Duration(seconds: 2), _maybeStartLocateLoop);
    }
  }

  void _nextPart(List<StepData> steps) {
    if (_currentStep + 1 >= steps.length) return;
    HapticFeedback.lightImpact();
    _stopLocateLoop();
    _resumeCamera();
    setState(() {
      _currentStep         = _currentStep + 1;
      _arState             = _ARState.scanning;
      _attemptCount        = 0;
      _panelExpanded       = false;
      _dynamicFeedback     = '';
      _cameraGuidance      = '';
      _partDescription     = '';
      _smoothBbox          = null;
      _prevBbox            = null;
      _stableFrameCount    = 0;
      _locateAttempts      = 0;
      _bboxLocked          = false;
      _partLocked          = false;
      _lastCorrectionSent  = null;
      _lastAiBbox          = null;
      _kfInitialised       = false;
      _kfPCx               = 1.0;
      _kfPCy               = 1.0;
      _netFailCount        = 0;
      _everDetected            = false;  // new part — no prior detection
      _lastGuidanceText        = '';
      _preDetectionHintFired   = false;
      _velCx               = 0.0;
      _velCy               = 0.0;
      _lastBboxTime        = null;
      _trackingEnabled     = false;
      _inspectionPanelVisible = false;
    });
    _attemptResults.clear();
    _verifiedCtrl.reset();
    _bboxFadeCtrl.reset();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final prov  = context.read<DiagnosisProvider>();
      final steps = prov.solution?.steps ?? _demoSteps;
      if (_currentStep < steps.length) {
        final s = steps[_currentStep];
        if (s.requiresDecisionPanel || s.isActionStep) {
          setState(() => _inspectionPanelVisible = true);
        } else {
          _maybeStartLocateLoop();
        }
      }
    });
  }

  // ── Inspection / observation panel ───────────────────────────────────────

  /// Farmer tapped an answer button in the inspection panel.
  void _onInspectionAnswer(StepOption option, List<StepData> steps) {
    HapticFeedback.mediumImpact();
    _attemptResults.add({
      'attempt_count': 1,
      'status':        'answered',
      'detected_part': option.id,
      'feedback':      option.labelEn,
    });
    setState(() => _inspectionPanelVisible = false);

    if (option.nextStep.isNotEmpty) {
      final prov = context.read<DiagnosisProvider>();
      final targetIdx = prov.solution?.indexOfStepId(option.nextStep) ?? -1;
      if (targetIdx >= 0 && targetIdx < steps.length) {
        Future.delayed(const Duration(milliseconds: 300), () {
          if (!mounted) return;
          _stopLocateLoop();
          setState(() {
            _currentStep            = targetIdx;
            _arState                = _ARState.scanning;
            _attemptCount           = 0;
            _panelExpanded          = false;
            _dynamicFeedback        = '';
            _cameraGuidance         = '';
            _smoothBbox             = null;
            _prevBbox               = null;
            _stableFrameCount       = 0;
            _locateAttempts         = 0;
            _bboxLocked             = false;
            _partLocked             = false;
            _lastCorrectionSent     = null;
            _lastAiBbox             = null;
            _kfInitialised          = false;
            _kfPCx                  = 1.0;
            _kfPCy                  = 1.0;
            _velCx                  = 0.0;
            _velCy                  = 0.0;
            _lastBboxTime           = null;
            _trackingEnabled        = false;
            _everDetected           = false;
            _lastGuidanceText       = '';
            _preDetectionHintFired  = false;
            _inspectionPanelVisible = false;
          });
          _attemptResults.clear();
          _verifiedCtrl.reset();
          _bboxFadeCtrl.reset();
          final nextS = steps[targetIdx];
          if (nextS.requiresDecisionPanel || nextS.isActionStep) {
            setState(() => _inspectionPanelVisible = true);
          } else {
            _maybeStartLocateLoop();
          }
        });
        return;
      }
    }

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

    // ── Orientation change → reset bbox immediately ───────────────────────
    // When the phone rotates, the old normalised bbox coords no longer map to
    // the correct screen positions. Reset smoothing so the arrow doesn't drift.
    final _curOrientation = MediaQuery.of(context).orientation;
    if (_lastOrientation != null && _lastOrientation != _curOrientation) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        setState(() {
          _smoothBbox       = null;
          _prevBbox         = null;
          _stableFrameCount = 0;
          _bboxLocked       = false;
          if (_arState == _ARState.guiding) _arState = _ARState.locating;
        });
        _stopTrackingTimer();
        _partLocked      = false;
        _lastCorrectionSent = null;
        _lastAiBbox      = null;
        _kfInitialised   = false;
        _kfPCx = 1.0; _kfPCy = 1.0;
        _velCx = 0; _velCy = 0; _lastBboxTime = null;
        _bboxFadeCtrl.reverse();
        if (_arState != _ARState.verified) _maybeStartLocateLoop();
      });
    }
    _lastOrientation = _curOrientation;

    // Compute the scan-box rect so the blur cut-out aligns with the painted box.
    final viewportH    = screenH - _kPanelHeight - safePad.top;
    const colTotal     = _kBoxSize + 28 + _kCaptureSize + 10 + 20.0;
    final colTopOffset = ((viewportH - colTotal) / 2).clamp(0.0, viewportH);
    // ADDED: Shift the entire AR assembly down by 50 pixels to clear the toast
    final double shiftDown = 50.0; 
    final boxTop       = safePad.top + colTopOffset + shiftDown;
    final boxLeft      = (screenW - _kBoxSize) / 2;
    final boxRect      = Rect.fromLTWH(boxLeft, boxTop, _kBoxSize, _kBoxSize);

    // FIX 10: Orientation-aware bbox→screen coordinate conversion.
    // The camera sensor on most Android phones is rotated 90° relative to the
    // display. CameraPreview handles the visual rotation internally, but we
    // must match the same logic when projecting the Gemini-returned bbox onto
    // screen pixels — otherwise the arrow drifts in portrait vs landscape.
    //
    // Rule: if the phone is in portrait and sensor is landscape-native (90/270°),
    // previewSize.width is the LONG sensor axis → displayed as the screen HEIGHT.
    // CameraPreview fills screen width, so the aspect ratio flips.
    final _camVal     = _cameraController?.value;
    final previewSize = _camVal?.previewSize;
    final previewW    = screenW;
    // Detect landscape mode via MediaQuery (portrait = default).
    final _isLandscape =
        MediaQuery.of(context).orientation == Orientation.landscape;
    final double previewH;
    if (previewSize == null) {
      previewH = screenH;
    } else if (_isLandscape) {
      // Landscape: sensor and display axes are aligned.
      previewH = screenW * (previewSize.height / previewSize.width);
    } else {
      // Portrait: sensor is rotated — swap width/height for aspect ratio.
      previewH = screenW * (previewSize.width / previewSize.height);
    }

    // Capture button label adapts to AR state
    final captureLabel = switch (_arState) {
      _ARState.guiding  => isHindi ? 'सत्यापित करें'     : 'Verify Part',
      _ARState.locating => isHindi ? 'भाग ढूंढ रहे हैं…' : 'Locating…',
      _              => isHindi ? 'भाग विश्लेषण करें' : 'Analyze Part',
    };

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

            // 3. AR Arrow overlay — drawn OVER camera, UNDER gradient
            // Only visible when guiding and bbox is stable
            // FIX 11: also render while locked (user tapped Verify — show locked arrow)
            if ((_arState == _ARState.guiding || _bboxLocked) && _smoothBbox != null)
              Positioned.fill(
                child: FadeTransition(
                  opacity: _bboxFadeAnim,
                  child: AnimatedBuilder(
                    animation: _arrowPulseCtrl,
                    builder: (_, __) => CustomPaint(
                      painter: _ARArrowPainter(
                        bbox:        _smoothBbox!,
                        previewW:    previewW,
                        previewH:    previewH,
                        pulseValue:  _arrowPulseCtrl.value,
                        partLabel:   _partDescription.isNotEmpty
                            ? _partDescription
                            : (step?.visualCue ?? '').replaceAll('_', ' '),
                        isHindi:     isHindi,
                      ),
                    ),
                  ),
                ),
              ),

            // 3b. Camera guidance chip when locating but part not found
            if (_arState == _ARState.locating && _cameraGuidance.isNotEmpty)
              Positioned(
                top: boxRect.top - 64,
                left: 24, right: 24,
                child: Center(
                  child: _CameraGuidanceChip(
                    message: _cameraGuidance,
                    isHindi: isHindi,
                  ),
                ),
              ),

            // 4. Top-bottom gradient tint
            _GradientOverlay(arState: _arState),

            // 5. Expanded panel scrim
            if (_panelExpanded)
              Positioned.fill(
                child: GestureDetector(
                  onTap: () => setState(() => _panelExpanded = false),
                  child: Container(color: Colors.black.withOpacity(0.60)),
                ),
              ),

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
                  captureLabel: captureLabel,
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
                arState:     _arState,
                isHindi:     isHindi,
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
                feedbackMsg: _cameraGuidance.isNotEmpty
                    ? _cameraGuidance
                    : _dynamicFeedback,
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
                    _transitionTo(_ARState.locating);
                    _maybeStartLocateLoop();
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
// Camera guidance chip — shown above scan box when part not in frame
// ─────────────────────────────────────────────────────────────────────────
class _CameraGuidanceChip extends StatelessWidget {
  final String message;
  final bool   isHindi;
  const _CameraGuidanceChip({required this.message, required this.isHindi});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.78),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
            color: _C.arrowBlue.withOpacity(0.55), width: 1.2)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.arrow_upward_rounded,
              color: _C.arrowBlue, size: 16),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              message,
              textAlign: TextAlign.center,
              style: GoogleFonts.inter(
                fontSize: 13, fontWeight: FontWeight.w600,
                color: _C.arrowBlue, height: 1.35)),
          ),
        ],
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
  final _ARState     arState;
  final bool         isHindi;
  final VoidCallback onClose;
  final VoidCallback onVoice;
  const _TopBar({
    required this.voiceActive,
    required this.arState,
    required this.isHindi,
    required this.onClose,
    required this.onVoice,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
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
        if (voiceActive)
          _SpeakingPill(isHindi: isHindi)
        else if (arState == _ARState.locating)
          _LocatingPill(isHindi: isHindi),
      ],
    );
  }
}

// Locating pill — shown in top bar during the AR locate loop
class _LocatingPill extends StatelessWidget {
  final bool isHindi;
  const _LocatingPill({required this.isHindi});
  @override
  Widget build(BuildContext context) {
    final label = isHindi ? 'भाग ढूंढ रहा है…' : 'Scanning for part…';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.60),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: _C.arrowBlue.withOpacity(0.35), width: 1)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 14, height: 14,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              valueColor: const AlwaysStoppedAnimation(_C.arrowBlue),
            ),
          ),
          const SizedBox(width: 7),
          Text(label,
            style: GoogleFonts.inter(
              fontSize: 12, fontWeight: FontWeight.w500,
              color: const Color(0xFFEAEAEA))),
        ],
      ),
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
  final String            captureLabel;
  final bool              isHindi;

  const _CentreArea({
    required this.arState,
    required this.verifiedFade,
    required this.onCapture,
    required this.boxRect,
    required this.captureLabel,
    required this.isHindi,
  });

  @override
  Widget build(BuildContext context) {
    final lblTarget = isHindi ? 'लक्ष्य' : 'TARGET';
    final boxColor = switch (arState) {
      _ARState.guiding  => _C.arrowGreen,
      _ARState.locating => _C.arrowBlue,
      _ARState.unclear  => _C.warning,
      _ARState.verified => _C.primary,
      _              => Colors.white,
    };
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
                  color: boxColor,
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
                  enabled: arState == _ARState.scanning ||
                           arState == _ARState.guiding  ||
                           arState == _ARState.unclear,
                ),
                const SizedBox(height: 10),
                AnimatedDefaultTextStyle(
                  duration: const Duration(milliseconds: 200),
                  style: GoogleFonts.inter(
                    fontSize: 17, fontWeight: FontWeight.w400,
                    color: (arState == _ARState.scanning ||
                            arState == _ARState.guiding)
                        ? _C.textSoft.withOpacity(0.85)
                        : _C.textMuted.withOpacity(0.40)),
                  child: Text(captureLabel),
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
            gradient: LinearGradient(
                colors: [_C.arrowGreen, _C.primary]),
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

              if (arState == _ARState.unclear) ...[
                const SizedBox(height: 10),
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 12, vertical: 10),
                  decoration: BoxDecoration(
                    color: _C.warning.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                        color: _C.warning.withOpacity(0.40), width: 1)),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Padding(
                        padding: EdgeInsets.only(top: 2),
                        child: Icon(Icons.photo_camera_rounded,
                            color: _C.warning, size: 16)),
                      const SizedBox(width: 8),
                      Expanded(child: Text(
                        feedbackMsg.isNotEmpty
                            ? feedbackMsg
                            : (isHindi
                                ? 'फोन स्थिर रखें और भाग फ्रेम में आए, फिर विश्लेषण करें।'
                                : 'Hold your phone steady and ensure the part fills the frame, then tap Analyze again.'),
                        style: GoogleFonts.inter(
                          fontSize: 13, fontWeight: FontWeight.w500,
                          color: _C.warning, height: 1.45))),
                    ],
                  ),
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

// ═══════════════════════════════════════════════════════════════════════════
// AR Arrow Painter
// Renders over the full camera preview (not clipped to scan box):
//   1. Pulsing bbox rectangle with corner brackets
//   2. Dashed arrow from screen bottom-centre to part centre
//   3. Instruction chip on arrow shaft
//
// All rendering is in screen pixels. bbox is normalised (0–1) converted here.
// Requires NO native AR/Unity SDK — pure Flutter CustomPainter.
// ═══════════════════════════════════════════════════════════════════════════
class _ARArrowPainter extends CustomPainter {
  final _NormBbox bbox;
  final double    previewW, previewH;
  final double    pulseValue;    // 0.0–1.0 from repeat(reverse:true) controller
  final String    partLabel;
  final bool      isHindi;

  static const _green = Color(0xFF22C55E);

  const _ARArrowPainter({
    required this.bbox,
    required this.previewW,
    required this.previewH,
    required this.pulseValue,
    required this.partLabel,
    required this.isHindi,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final partRect   = bbox.toScreenRect(previewW, previewH);

    // ── 1. Bbox fill + border ─────────────────────────────────────────────
    final alpha    = 0.55 + 0.30 * pulseValue;
    final fillPaint = Paint()
      ..color = _green.withOpacity(0.18 + 0.10 * pulseValue)
      ..style = PaintingStyle.fill;
    final borderPaint = Paint()
      ..color       = _green.withOpacity(alpha)
      ..style       = PaintingStyle.stroke
      ..strokeWidth = 4.0;

    final rRect = RRect.fromRectAndRadius(partRect, const Radius.circular(10));
    canvas.drawRRect(rRect, fillPaint);
    // White shadow behind green — ensures visibility on any bg color
    final shadowPaint = Paint()
      ..color       = Colors.white.withOpacity(0.90)
      ..style       = PaintingStyle.stroke
      ..strokeWidth = 6.5;  // slightly wider than green border
    canvas.drawRRect(rRect, shadowPaint);
    canvas.drawRRect(rRect, borderPaint);

    // ── 2. Corner bracket accents ─────────────────────────────────────────
    _drawCornerBrackets(
        canvas, partRect, _green.withOpacity(0.85 + 0.15 * pulseValue));

    // ── 3. Dashed arrow ───────────────────────────────────────────────────
    // Tail: bottom-centre of screen, just above bottom panel
    final tail = Offset(size.width / 2, size.height * 0.72);
    // Head: nearest edge of bbox toward the tail
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

    _drawDashedLine(canvas, tail, head, arrowShadow);  // white shadow first
    _drawDashedLine(canvas, tail, head, arrowPaint);   // green on top
    _drawArrowhead(canvas, tail, head);

    // ── 4. Instruction chip on shaft ──────────────────────────────────────
    if (partLabel.isNotEmpty) {
      final mid = Offset(
        (tail.dx + head.dx) / 2,
        (tail.dy + head.dy) / 2,
      );
      _drawChip(canvas, size, mid, partLabel);
    }
  }

  void _drawCornerBrackets(Canvas canvas, Rect rect, Color color) {
    // White shadow bracket underneath green — high contrast on any surface
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
    const L = 28.0;  // larger brackets — easier to see on textured surfaces
    final corners = [
      [Offset(rect.left, rect.top + L),      Offset(rect.left, rect.top),      Offset(rect.left  + L, rect.top)],
      [Offset(rect.right - L, rect.top),     Offset(rect.right, rect.top),     Offset(rect.right,     rect.top + L)],
      [Offset(rect.left, rect.bottom - L),   Offset(rect.left, rect.bottom),   Offset(rect.left  + L, rect.bottom)],
      [Offset(rect.right - L, rect.bottom),  Offset(rect.right, rect.bottom),  Offset(rect.right,     rect.bottom - L)],
    ];
    // Shadow pass
    for (final c in corners) {
      canvas.drawPath(
        Path()
          ..moveTo(c[0].dx, c[0].dy)
          ..lineTo(c[1].dx, c[1].dy)
          ..lineTo(c[2].dx, c[2].dy),
        shadow,
      );
    }
    // Green pass on top
    for (final c in corners) {
      canvas.drawPath(
        Path()
          ..moveTo(c[0].dx, c[0].dy)
          ..lineTo(c[1].dx, c[1].dy)
          ..lineTo(c[2].dx, c[2].dy),
        p,
      );
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
    final display = label.length > 28 ? '{label.substring(0, 25)}…' : label;
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

    canvas.drawRRect(r,
        Paint()..color = Colors.black.withOpacity(0.72));
    canvas.drawRRect(r,
        Paint()
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
    final tx = nx != 0 ? (nx > 0 ? rect.right - cx : rect.left - cx) / nx : double.infinity;
    final ty = ny != 0 ? (ny > 0 ? rect.bottom - cy : rect.top - cy) / ny : double.infinity;
    final t  = math.min(tx.abs(), ty.abs());
    return Offset(cx + nx * t, cy + ny * t);
  }

  @override
  bool shouldRepaint(_ARArrowPainter old) =>
      old.bbox       != bbox       ||
      old.pulseValue != pulseValue ||
      old.partLabel  != partLabel;
}

// ─────────────────────────────────────────────────────────────────────────
// Demo steps fallback
// ─────────────────────────────────────────────────────────────────────────
final _demoSteps = List.generate(12, (i) => StepData(
  stepId:      's{i + 1}',
  stepType:    i == 2 ? StepType.inspection
             : i == 6 ? StepType.action
             : i == 9 ? StepType.observation
             : StepType.visual,
  stepTitleEn: 'Step {i + 1}',
  stepTitleHi: 'चरण {i + 1}',
  text:      'Inspect component {i + 1} carefully before proceeding.',
  textEn:    'Inspect component {i + 1} carefully before proceeding.',
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
    StepOption(id: 'd', labelEn: 'Haven\'t checked yet', labelHi: 'अभी नहीं देखा', nextStep: 's3'),
  ] : [],
));