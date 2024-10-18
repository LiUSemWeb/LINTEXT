import 'package:flutter/material.dart';
import 'package:rel_ex_interface/src/util/json.dart';

// import 'settings_controller.dart';

// final selectedType = ValueNotifier<String?>(null);
// final selectedEnt = ValueNotifier(-1);

// Turn this into a "Highlight Criteria" object.
// Include "highlight temporary" and "remove temporary" to set values
// only while hovering.
// Then other classes can set highlights, such as when adding domain/range.
// Also allow for a list of types and wildcards for ents?
// Basically, turn this into a matcher instead of just equality.

class HoverFeatures with ChangeNotifier {
  String? selectedType;
  int ent = -1;

  void setType(String? newtype) {
    selectedType = newtype;
    notifyListeners();
  }

  void setEnt(int newEnt) {
    ent = newEnt;
    notifyListeners();
  }
}

class TokensView extends StatelessWidget {
  const TokensView(
      {super.key,
      required this.tokens,
      this.ent = -1,
      this.eType,
      this.showHover = true,
      this.selectedEnt = -1});
  // final String text;
  // final Color innerColor;
  // final Color outlineColor;
  final List<String> tokens;
  final int ent;
  final int selectedEnt;
  final String? eType;
  final bool showHover;
  // static int pos = 0;
  static final selected = HoverFeatures();
  static final Map<String, Color> typeMap = <String, Color>{};
  static final Map<int, Color> entityMap = <int, Color>{};

  String get text {
    StringBuffer sb = StringBuffer(tokens[0]);
    for (String t in tokens.skip(1)) {
      if (t.startsWith("##")) {
        sb.write(t.substring(2));
      } else {
        sb.write(' ');
        sb.write(t);
      }
    }
    return sb.toString();
  }

  String get tooltip {
    return "";
  }

  // Color get innerColor {
  //   return resolveEntityColor(ent);
  // }

  // Color get outlineColor {
  //   return resolveTypeColor(eType);
  // }

  static void resetColors() {
    typeMap.clear();
    entityMap.clear();
  }

  static Color resolveTypeColor(String? type) {
    if (type == null) return Colors.transparent;
    if (typeMap.containsKey(type)) {
      return typeMap[type]!;
    } else {
      return typeMap[type] =
          Colors.accents[(typeMap.length * 2) % Colors.accents.length];
    }
  }

  static Color resolveEntityColor(int ent) {
    if (ent < 0) return Colors.transparent;
    if (entityMap.containsKey(ent)) {
      return entityMap[ent]!;
    } else {
      return entityMap[ent] = Colors.primaries[Colors.primaries.length -
          ((entityMap.length) % Colors.primaries.length) -
          1];
    }
  }

  static List<Widget> fromJson(JSONObject json) {
    // print(json);
    if (json.isEmpty || !json.containsKey('tokens')) return [];

    List<Widget> outList = [];
    List<String> inProgress = [];
    String typeInProgress = "";
    int entInProgress = -1;
    for (JSONObject token in json['tokens']) {
      String text = token['text'];
      // String type = ;
      int ent = token['ent'];

      // An entity in progress
      if (entInProgress >= 0) {
        // And we've gone one past it.
        if (ent != entInProgress) {
          outList.add(
            TokensView(
              tokens: inProgress,
              eType: typeInProgress,
              ent: entInProgress,
              selectedEnt: selected.ent,
              // origTokens: inProgress,
            ),
          );
          inProgress = [];
          // We're still reading the entity.
        }
        // No entity in progress
      }
      if (ent >= 0) {
        inProgress.add(text);
        typeInProgress = token['type'];
      } else {
        outList.add(TokensView(tokens: [text]));
      }
      entInProgress = ent;
    }
    return outList;
  }

  static Set<String> typeListFromJson(JSONObject json) {
    Set<String> types = {};
    if (json.isEmpty || !json.containsKey('tokens')) return types;
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
          tokens: [t],
          eType: t,
          showHover: false,
        ));
      }
    }
    return outList;
  }

  Color getInnerColor(int other) {
    // int other = selectedEnt;
    double lerp = (other < 0)
        ? 0.2
        : (ent != other)
            ? 0.6
            : .0;
    return Color.lerp(
            TokensView.resolveEntityColor(ent), Colors.transparent, lerp) ??
        Colors.transparent;
  }

  Color getOutlineColor(String? other) {
    double lerp = (other == null)
        ? 0.2
        : (eType != other)
            ? 0.6
            : .0;
    return Color.lerp(
            TokensView.resolveTypeColor(eType), Colors.transparent, lerp) ??
        Colors.transparent;
    // return TokensView.resolveTypeColor(eType);
  }

  Color get outlineColor {
    return TokensView.resolveTypeColor(eType);
  }

  Color getTextColor(int other, BuildContext context) {
    Color innerColor = getInnerColor(other);
    return innerColor == Colors.transparent
        ? Theme.of(context).textTheme.bodyLarge?.color ?? Colors.black
        : ThemeData.estimateBrightnessForColor(innerColor) == Brightness.dark
            ? Colors.white
            : Colors.black;
  }

  @override
  Widget build(BuildContext context) {
    // Color textColor = innerColor == Colors.transparent
    //     ? Theme.of(context).textTheme.bodyLarge?.color ?? Colors.black
    //     : ThemeData.estimateBrightnessForColor(innerColor) == Brightness.dark
    //         ? Colors.white
    //         : Colors.black;

    Widget body = ListenableBuilder(
      listenable: selected,
      builder: (BuildContext context, Widget? child) {
        return Container(
          margin: const EdgeInsets.symmetric(horizontal: 1.0),
          decoration: BoxDecoration(
            // border: Border.all(color: color, width: 4),
            border: Border.all(
                color: getOutlineColor(selected.selectedType), width: 2
                // horizontal: BorderSide(color: outlineColor, width: 2),
                ),
            borderRadius: BorderRadius.circular(20),
            color: getInnerColor(selected.ent),
          ),
          child: Text(
            text,
            textAlign: TextAlign.center,
            style: TextStyle(color: getTextColor(selected.ent, context)),
          ),
        );
      },
    );

    if (showHover) {
      body = Tooltip(
        message: showHover
            ? ent >= 0
                ? 'Entity: $ent\nType: $eType\nTokens: ${tokens.join(" ")}'
                : 'Tokens: ${tokens.join(" ")}'
            : null,
        child: body,
      );
    }

    // if (entListener != null) {
    if (ent >= 0) {
      body = MouseRegion(
        child: body,
        onEnter: (event) {
          selected.setEnt(ent);
          // selectedEnt.value = ent;
          selected.selectedType = eType;
        },
        onExit: (event) {
          selected.setEnt(-1);
          selected.selectedType = null;
          // selectedType.value = null;
        },
      );
    }
    //     },
    //   );
    // }

    // body = ;

    return Wrap(children: [body]);
  }
}
