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

  TextStyle textStyleFromScore(double score, BuildContext context,
      {Color scorelessColor = Colors.transparent}) {
    Color? scoreColor = score >= 0.0
        ? Color.lerp(lowScoreColor, highScoreColor, score)
        : scorelessColor;

    return TextStyle(
        backgroundColor: scoreColor, color: getColor(scoreColor, context));
  }

  Widget buildInnerText(BuildContext context, int i) {
    String tx = text[i];
    double leftPad = 2.0;
    double rightPad = 2.0;
    if (tx.startsWith("##")) {
      tx = tx.substring(2);
      leftPad = 0.0;
    }
    if (text.length > (i + 1) && text[i + 1].startsWith("##")) {
      rightPad = 0.0;
    }

    Widget innerText = Text(
      tx,
      style: textStyleFromScore(scores[i], context),
    );
    if (scores[i] >= 0.0) {
      innerText = Tooltip(
        message: 'Token: ${text[i]}\nScore: ${scores[i].toStringAsFixed(3)}',
        child: innerText,
      );
    }
    return Padding(
      padding: EdgeInsets.only(left: leftPad, right: rightPad),
      child: innerText,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Wrap(
      children: [
        Row(children: [
          for (int i = 0; i < text.length; i++) buildInnerText(context, i)
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
