import 'package:flutter/material.dart';

class StepData {
  final String text;
  final String textEn;
  final String textHi;
  final String? visualCue;
  final String? arModel;
  final String? safetyWarning;

  StepData({
    required this.text,
    required this.textEn,
    required this.textHi,
    this.visualCue,
    this.arModel,
    this.safetyWarning,
  });

  factory StepData.fromJson(Map<String, dynamic> json) => StepData(
        text:          json['text']    ?? '',
        textEn:        json['text_en'] ?? json['text'] ?? '',
        textHi:        json['text_hi'] ?? json['text'] ?? '',
        visualCue:     json['visual_cue'],
        arModel:       json['ar_model'],
        safetyWarning: json['safety_warning'],
      );

  String getLocalizedText(bool isHindi) =>
      isHindi && textHi.isNotEmpty ? textHi : textEn.isNotEmpty ? textEn : text;
}

class SolutionData {
  final String machineType;
  final String problemIdentified;
  final List<StepData> steps;
  final List<String> safetyWarnings;
  final List<String> toolsNeeded;

  SolutionData({
    required this.machineType,
    required this.problemIdentified,
    required this.steps,
    required this.safetyWarnings,
    required this.toolsNeeded,
  });

  factory SolutionData.fromJson(Map<String, dynamic> json) => SolutionData(
        machineType:       json['machine_type'] ?? 'Agricultural Machine',
        problemIdentified: json['problem_identified'] ?? '',
        steps: (json['steps'] as List? ?? [])
            .map((s) => StepData.fromJson(s))
            .toList(),
        safetyWarnings: List<String>.from(json['safety_warnings_en'] ?? []),
        toolsNeeded:    List<String>.from(json['tools_needed'] ?? []),
      );
}

class DiagnosisProvider extends ChangeNotifier {
  Map<String, dynamic>? diagnosis;
  SolutionData? _solution;
  String?       _problemDescription;
  bool          _isLoading = false;
  String?       _error;

  SolutionData? get solution           => _solution;
  String?       get problemDescription => _problemDescription;
  bool          get isLoading          => _isLoading;
  String?       get error              => _error;
  bool          get hasSolution        => _solution != null;

  void setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }

  void setDiagnosis(Map<String, dynamic> json) {
    _solution            = SolutionData.fromJson(json['solution'] ?? json);
    _problemDescription  = json['problem_description'];
    _error               = null;
    _isLoading           = false;
    notifyListeners();
  }

  void setError(String error) {
    _error     = error;
    _isLoading = false;
    notifyListeners();
  }

  void clear() {
    _solution = null;
    _error    = null;
    notifyListeners();
  }
}
