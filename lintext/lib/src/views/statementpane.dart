import 'package:flutter/material.dart';

// import 'settings_controller.dart';

class StatementView extends StatelessWidget {
  const StatementView({
    super.key,
    required this.text,
    required this.scores,
    this.lowScoreColor = Colors.red,
    this.highScoreColor = Colors.green,
  });
  final List<String> text;
  final List<dynamic> scores;
  final Color lowScoreColor;
  final Color highScoreColor;

  Color? getColor(Color? innerColor, BuildContext context) {
    if (innerColor == null) return Theme.of(context).textTheme.bodyLarge?.color;
    Color textColor = innerColor == Colors.transparent
        ? Theme.of(context).textTheme.bodyLarge?.color ?? Colors.black
        : ThemeData.estimateBrightnessForColor(innerColor) == Brightness.dark
            ? Colors.white
            : Colors.black;
    return textColor;
  }

  @override
  Widget build(BuildContext context) {
    return Wrap(
      children: [
        Row(children: [
          for (int i = 0; i < text.length; i++)
            Padding(
              padding: const EdgeInsets.only(left: 2.0, right: 2.0),
              child: Text(
                text[i],
                style: TextStyle(
                    backgroundColor: Color.lerp(
                        lowScoreColor, highScoreColor, scores[i] as double),
                    color: getColor(
                        Color.lerp(
                            lowScoreColor, highScoreColor, scores[i] as double),
                        context)),
              ),
            )
        ]

            //  [
            //   Text(
            //     text,
            //     textAlign: TextAlign.center,
            //     style: TextStyle(color: textColor),
            //   ),
            // ],
            ),
      ],
    );
  }
}
