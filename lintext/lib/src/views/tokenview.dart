import 'package:flutter/material.dart';
import 'package:rel_ex_interface/src/util/json.dart';

// import 'settings_controller.dart';

class TokensView extends StatelessWidget {
  const TokensView(
      {super.key,
      required this.text,
      required this.innerColor,
      required this.outlineColor});
  final String text;
  final Color innerColor;
  final Color outlineColor;
  static int pos = 0;
  static final Map<String, Color> typeMap = <String, Color>{};
  static final Map<int, Color> entityMap = <int, Color>{};

  static void resetColors() {
    typeMap.clear();
    entityMap.clear();
  }

  static List<TokensView> fromJson(JSONObject json) {
    // print(json);
    if(json.isEmpty || !json.containsKey('tokens')) return [];

    List<TokensView> outList = [];
    for (JSONObject token in json['tokens']) {
      String text = token['text'];
      String type = token['type'];
      int ent = token['ent'];
      Color typeCol;
      if (ent >= 0) {
        if (typeMap.containsKey(type)) {
          typeCol = typeMap[type]!;
        } else {
          typeCol =
              Colors.accents[(typeMap.length * 2) % Colors.accents.length];
          typeMap[type] = typeCol;
        }
      } else {
        // col = Null;
        typeCol = Colors.transparent;
      }
      Color entCol;
      if (ent >= 0) {
        if (entityMap.containsKey(ent)) {
          entCol = entityMap[ent]!;
        } else {
          entCol = Colors.primaries[Colors.primaries.length -
              ((entityMap.length) % Colors.primaries.length) -
              1];
          entityMap[ent] = entCol;
        }
      } else {
        entCol = Colors.transparent;
      }
      outList.add(TokensView(
        text: text,
        innerColor: entCol,
        outlineColor: typeCol,
      ));
    }
    // print(typeMap);
    return outList;
  }

  static Set<String> typeListFromJson(JSONObject json) {
    Set<String> types = {};
    if(json.isEmpty || !json.containsKey('tokens')) return types;
    for (JSONObject token in json['tokens']) {
      types.add(token['type']);
    }
    return types;
  }

  static List<TokensView> typeListWidgetFromJson(JSONObject json) {
    List<TokensView> outList = [];
    for (String t in typeListFromJson(json)) {
      if (t.isNotEmpty) {
        outList.add(TokensView(
          text: t,
          innerColor: Colors.transparent,
          outlineColor: typeMap[t]!,
        ));
      }
    }
    return outList;
  }

  @override
  Widget build(BuildContext context) {
    Color textColor = innerColor == Colors.transparent
        ? Theme.of(context).textTheme.bodyLarge?.color ?? Colors.black
        : ThemeData.estimateBrightnessForColor(innerColor) == Brightness.dark
            ? Colors.white
            : Colors.black;

    return Wrap(
      children: [
        Container(
          margin: const EdgeInsets.symmetric(horizontal: 1.0),
          decoration: BoxDecoration(
            // border: Border.all(color: color, width: 4),
            border: Border.symmetric(
              vertical: BorderSide(color: outlineColor, width: 2),
              horizontal: BorderSide(color: outlineColor, width: 2),
            ),
            borderRadius: BorderRadius.circular(20),
            color: innerColor,
          ),
          child: Text(
            text,
            textAlign: TextAlign.center,
            style: TextStyle(color: textColor),
          ),
        ),
      ],
    );
  }
}
