import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

// ─────────────────────────────────────────────────────────────────────────────
// Exceptions
// ─────────────────────────────────────────────────────────────────────────────
class RequestTimeoutException implements Exception {
  final String message;
  const RequestTimeoutException([this.message = 'Request timed out']);
  @override
  String toString() => message;
}

class NetworkException implements Exception {
  final String message;
  const NetworkException(this.message);
  @override
  String toString() => message;
}

class ServerException implements Exception {
  final int statusCode;
  final String message;
  const ServerException(this.statusCode, this.message);
  @override
  String toString() => 'ServerException($statusCode): $message';
}

// ─────────────────────────────────────────────────────────────────────────────
// Result type passed to DiagnosisProvider
// ─────────────────────────────────────────────────────────────────────────────
class DiagnosisResult {
  final Map<String, dynamic> raw;
  DiagnosisResult(this.raw);
}

// ─────────────────────────────────────────────────────────────────────────────
// ApiService
// ─────────────────────────────────────────────────────────────────────────────
class ApiService {
  static const String _baseUrl = 
      'https://technospes-agrifixar-backend-new.hf.space';
      // 'http://10.0.2.2:7860';
  static const String _appKey = '020b082f133f403abf8694e6144df1a79396b2706dd9de108bc54a05e891fc29';
  static const Duration _uploadTimeout = Duration(seconds: 90);
  static const Duration _streamTimeout = Duration(minutes: 3);
  static const int _maxRetries = 2;

  // ── Original batch endpoint ─────────────────────────────────────────────
  static Future<DiagnosisResult> uploadAndDiagnose({
    required String videoPath,
    required String audioPath,
    String machineType = 'tractor',
    String language = 'en',
    void Function(double progress)? onProgress,
    void Function(String status)? onStatus,
  }) async {
    if (!await File(videoPath).exists()) {
      throw NetworkException('Video file not found: $videoPath');
    }
    if (!await File(audioPath).exists()) {
      throw NetworkException('Audio file not found: $audioPath');
    }

    onStatus?.call('Preparing files…');
    onProgress?.call(0.05);

    final request = http.MultipartRequest('POST', Uri.parse('$_baseUrl/diagnose'));
    request.headers['X-App-Key'] = _appKey;
    request.fields['machine_type'] = machineType;
    request.fields['language'] = language;
    request.files.add(await http.MultipartFile.fromPath('video', videoPath));
    request.files.add(await http.MultipartFile.fromPath('audio', audioPath));

    onStatus?.call('Uploading to server…');
    onProgress?.call(0.15);

    return _sendWithRetry(
      request: request,
      timeout: _uploadTimeout,
      onRetry: (attempt) {
        onStatus?.call('Retrying… (attempt $attempt)');
        onProgress?.call(0.15);
      },
      onSuccess: (json) {
        onStatus?.call('Solution ready!');
        onProgress?.call(1.0);
        return DiagnosisResult(json);
      },
    );
  }

  // ── Streaming SSE endpoint with retry capability ─────────────────────────
  static Future<DiagnosisResult> uploadAndDiagnoseStreaming({
    required String videoPath,
    required String audioPath,
    required void Function(int stageIndex) onStageStart,
    required void Function(int stageIndex, Map<String, dynamic> data) onStageComplete,
    String machineType = 'tractor',
    String language = 'en',
    void Function(double progress)? onUploadProgress,
    void Function(String status)? onUploadStatus,
  }) async {
    if (!await File(videoPath).exists()) {
      throw NetworkException('Video file not found: $videoPath');
    }
    if (!await File(audioPath).exists()) {
      throw NetworkException('Audio file not found: $audioPath');
    }

    onUploadStatus?.call('Preparing files…');
    onUploadProgress?.call(0.05);

    final request = http.MultipartRequest(
        'POST', Uri.parse('$_baseUrl/diagnose/stream'));
    request.headers['X-App-Key'] = _appKey;
    request.fields['machine_type'] = machineType;
    request.fields['language'] = language;
    request.files.add(await http.MultipartFile.fromPath('video', videoPath));
    request.files.add(await http.MultipartFile.fromPath('audio', audioPath));

    onUploadStatus?.call('Uploading to server…');
    onUploadProgress?.call(0.15);

    return _sendStreamWithRetry(
      request: request,
      timeout: _streamTimeout,
      onRetry: (attempt) {
        onUploadStatus?.call('Retrying upload… (attempt $attempt)');
        onUploadProgress?.call(0.15);
      },
      onStageStart: onStageStart,
      onStageComplete: onStageComplete,
      onUploadProgress: onUploadProgress,
      onUploadStatus: onUploadStatus,
    );
  }

  // ── Internal: send with timeout + exponential back-off retry ─────────────
  static Future<DiagnosisResult> _sendWithRetry({
    required http.MultipartRequest request,
    required Duration timeout,
    int maxRetries = _maxRetries,
    void Function(int attempt)? onRetry,
    required DiagnosisResult Function(Map<String, dynamic>) onSuccess,
  }) async {
    int attempt = 0;

    while (true) {
      try {
        final clone = _clone(request);
        final streamed = await clone.send().timeout(
          timeout,
          onTimeout: () { throw const RequestTimeoutException(); },
        );
        final body = await streamed.stream.bytesToString();

        if (streamed.statusCode == 200) {
          try {
            final json = jsonDecode(body) as Map<String, dynamic>;
            return onSuccess(json);
          } catch (_) {
            throw ServerException(200, 'Response was not valid JSON: $body');
          }
        }

        if (streamed.statusCode >= 500 && attempt < maxRetries) {
          attempt++;
          onRetry?.call(attempt);
          await Future.delayed(Duration(seconds: attempt * 2));
          continue;
        }

        throw ServerException(streamed.statusCode, body);

      } on RequestTimeoutException {
        rethrow;

      } on SocketException catch (e) {
        if (attempt < maxRetries) {
          attempt++;
          onRetry?.call(attempt);
          await Future.delayed(Duration(seconds: attempt));
          continue;
        }
        throw NetworkException('No internet connection: ${e.message}');

      } on http.ClientException catch (e) {
        throw NetworkException('Connection error: ${e.message}');

      } catch (e) {
        if (e is RequestTimeoutException ||
            e is NetworkException ||
            e is ServerException) rethrow;
        throw NetworkException('Unexpected error: $e');
      }
    }
  }

  // ── Internal: streaming version with retry capability ────────────────────
  static Future<DiagnosisResult> _sendStreamWithRetry({
    required http.MultipartRequest request,
    required Duration timeout,
    int maxRetries = _maxRetries,
    void Function(int attempt)? onRetry,
    required void Function(int stageIndex) onStageStart,
    required void Function(int stageIndex, Map<String, dynamic> data) onStageComplete,
    void Function(double progress)? onUploadProgress,
    void Function(String status)? onUploadStatus,
  }) async {
    int attempt = 0;

    while (true) {
      final client = http.Client();
      DiagnosisResult? finalResult;
      
      try {
        final clone = _clone(request);
        final streamed = await client.send(clone).timeout(
          timeout,
          onTimeout: () { throw const RequestTimeoutException(); },
        );

        if (streamed.statusCode != 200) {
          final body = await streamed.stream.bytesToString();
          
          if (streamed.statusCode >= 500 && attempt < maxRetries) {
            attempt++;
            onRetry?.call(attempt);
            client.close();
            await Future.delayed(Duration(seconds: attempt * 2));
            continue;
          }
          
          throw ServerException(streamed.statusCode, body);
        }

        onUploadStatus?.call('Processing diagnosis…');
        onUploadProgress?.call(0.3);

        final buffer = StringBuffer();
        await for (final chunk in streamed.stream.transform(utf8.decoder)) {
          buffer.write(chunk);
          final raw = buffer.toString();
          final events = raw.split('\n\n');
          buffer.clear();
          buffer.write(events.last);

          for (int i = 0; i < events.length - 1; i++) {
            final eventText = events[i].trim();
            if (eventText.isEmpty) continue;
            
            final jsonStr = eventText.startsWith('data: ') 
                ? eventText.substring(6).trim() 
                : eventText.trim();
            if (jsonStr.isEmpty) continue;

            final Map<String, dynamic> parsed;
            try { 
              parsed = json.decode(jsonStr) as Map<String, dynamic>; 
            } catch (_) { 
              continue; 
            }

            final eventType = parsed['event'] as String?;
            final stage = parsed['stage'] as int?;

            if (stage != null) {
              final progress = 0.3 + (stage * 0.2);
              onUploadProgress?.call(progress.clamp(0.3, 0.9));
            }

            switch (eventType) {
              case 'stage_start':
                if (stage != null) {
                  onUploadStatus?.call('Stage ${stage + 1}/4 starting…');
                  onStageStart(stage);
                }
                break;

              case 'stage_done':
                if (stage != null) {
                  onUploadStatus?.call('Stage ${stage + 1}/4 completed');
                  onStageComplete(stage, parsed);
                  if (stage == 3 && parsed.containsKey('result')) {
                    finalResult = DiagnosisResult(parsed['result'] as Map<String, dynamic>);
                    onUploadStatus?.call('Diagnosis complete!');
                    onUploadProgress?.call(1.0);
                  }
                }
                break;

              case 'error':
                final msg = parsed['message'] as String? ?? 'Unknown backend error';
                throw Exception(msg);
            }
          }
        }

        if (finalResult == null) {
          throw const NetworkException('Stream ended without a diagnosis result. Check backend logs.');
        }
        
        return finalResult;

      } on TimeoutException {
        rethrow;
        
      } on SocketException catch (e) {
        if (attempt < maxRetries) {
          attempt++;
          onRetry?.call(attempt);
          client.close();
          await Future.delayed(Duration(seconds: attempt));
          continue;
        }
        throw NetworkException('Connection lost during streaming: ${e.message}');
        
      } on http.ClientException catch (e) {
        if (attempt < maxRetries) {
          attempt++;
          onRetry?.call(attempt);
          client.close();
          await Future.delayed(Duration(seconds: attempt));
          continue;
        }
        throw NetworkException('Stream connection error: ${e.message}');
        
      } catch (e) {
        if (e is RequestTimeoutException ||
            e is NetworkException ||
            e is ServerException) rethrow;
        throw NetworkException('Unexpected streaming error: $e');
        
      } finally {
        client.close();
      }
    }
  }

  static http.MultipartRequest _clone(http.MultipartRequest src) {
    final dst = http.MultipartRequest(src.method, src.url);
    dst.headers.addAll(src.headers);
    dst.fields.addAll(src.fields);
    for (final f in src.files) dst.files.add(f);
    return dst;
  }

  // ── Verify a repair step photo ────────────────────────────────────────────
  static Future<Map<String, dynamic>> verifyStep({
    required File imageFile,
    required String stepText,
    required String machineType,
    required String problemContext,
    required int attemptCount,
    String previousSteps = '[]',                  // ← visual memory from client
    void Function(double progress)? onProgress,
    void Function(String status)? onStatus,
  }) async {
    final request = http.MultipartRequest(
        'POST', Uri.parse('$_baseUrl/verify_step'))
      ..headers['X-App-Key'] = _appKey
      ..files.add(await http.MultipartFile.fromPath('image', imageFile.path))
      ..fields.addAll({
        'step_text':       stepText,
        'machine_type':    machineType,
        'problem_context': problemContext,
        'attempt_count':   attemptCount.toString(),
        'previous_steps':  previousSteps,          // ← was hardcoded '[]'
      });

    onStatus?.call('Verifying step…');
    onProgress?.call(0.5);

    final result = await _sendWithRetry(
      request: request,
      timeout: const Duration(seconds: 30),
      maxRetries: 1,
      onRetry: (attempt) {
        onStatus?.call('Retrying verification… (attempt $attempt)');
        onProgress?.call(0.5);
      },
      onSuccess: (json) => DiagnosisResult(json), // Fixed: return DiagnosisResult
    );

    onStatus?.call('Verification complete');
    onProgress?.call(1.0);
    return result.raw; // Fixed: return the raw map
  }
}