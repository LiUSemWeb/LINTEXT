import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

class AnalyzeView extends StatelessWidget {
  const AnalyzeView({
    super.key,
    required this.callback,
  });

  final AsyncCallback callback;

  @override
  Widget build(BuildContext context) {
    return TextButton(onPressed: callback, child: const Text('Analyze!'),);
  }
}