// ignore_for_file: type_literal_in_constant_pattern

import 'dart:convert';
import 'dart:io';
// import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
// import 'package:rel_ex_interface/src/views/mainpane.dart';

import '../util/dir.dart';
import '../util/json.dart';

bool validateTypes = true;

// stores ExpansionPanel state information
class SchemaItem extends StatefulWidget {
  const SchemaItem({super.key, required this.json, required this.typeHints});
  final JSONObject json;
  final Set<String> typeHints;

  @override
  State<SchemaItem> createState() => _SchemaItemState();

  static Map<String, String> labelNames = {
    'rel_id': 'Relation identifier',
    'name': 'Relation name',
    'desc': 'Description',
    'prompt_xy': 'Prompt (Subject first)',
    'prompt_yx': 'Prompt (Object first)',
    'domain': 'Domain',
    'range': 'Range',
    'reflexive': 'Reflexive',
    'irreflexive': 'Irreflexive',
    'symmetric': 'Symmetric',
    'antisymmetric': 'Antisymmetric',
    'transitive': 'Transitive',
    'implied_by': 'Implied by (Follows from)',
    'tokens': 'Tokenized prompt',
    'verb': 'Verb position'
  };

  // static SchemaItem fromJson(JSONObject json) {
  //   return SchemaItem(
  //     // rel_id: json['rel_id'],
  //     // name: json['name'],
  //     // desc: json['desc'],
  //     // prompt_xy: json['prompt_xy'],
  //     // prompt_yx: json['prompt_yx'],
  //     // domain: json['domain'],
  //     // range: json['range'],
  //     // reflexive: json['reflexive'],
  //     // irreflexive: json['irreflexive'],
  //     // symmetric: json['symmetric'],
  //     // antisymmetric: json['antisymmetric'],
  //     // transitive: json['transitive'],
  //     // implied_by: json['implied_by'],
  //     // tokens: json['tokens'],
  //     // verb: json['verb'],
  //     json: json,
  //     typeHints: Schem,
  //   );
  // }

  JSONObject toJSON() {
    return json;
    // return {
    //   'rel_id': rel_id,
    //   'name': name,
    //   'desc': desc,
    //   'prompt_xy': prompt_xy,
    //   'prompt_yx': prompt_yx,
    //   'domain': domain,
    //   'range': range,
    //   'reflexive': reflexive,
    //   'irreflexive': irreflexive,
    //   'symmetric': symmetric,
    //   'antisymmetric': antisymmetric,
    //   'transitive': transitive,
    //   'implied_by': implied_by,
    //   'tokens': tokens,
    //   'verb': verb,
    // };
  }
}

class _SchemaItemState extends State<SchemaItem> {
  // late final List<SchemaItem> _data;
  late Map<String, TextEditingController> controllers;
  bool isExpanded = false;

  @override
  void initState() {
    // print("schemaJSON: ${widget.schemaJson}");
    // TODO: Set default value here by making a method and
    // using case-based logic around the variable type.
    controllers = widget.json.map(
        (k, v) => MapEntry(k, TextEditingController(text: getInitialValue(k))));
    super.initState();
  }

  String getInitialValue(String key) {
    dynamic val = widget.json[key];
    if (key == 'domain' || key == 'range') {
      return val.join(', ');
    }

    // Type t = ;
    switch (val.runtimeType) {
      case String:
      case int:
      case bool:
        return val.toString();
      case List:
        return val.join(', ');
      default:
        return "Unknown type: ${val.runtimeType}";
    }
  }

  @override
  void didUpdateWidget(covariant SchemaItem oldWidget) {
    super.didUpdateWidget(oldWidget);
  }

  void updateString(String key, String value) {
    widget.json[key] = value;
  }

  void updateList() {}

  @override
  Widget build(BuildContext context) {
    TextStyle ts = TextStyle(color: Colors.primaries[1]);
    List<Widget> children = [];

    for (String key in ['rel_id', 'name', 'desc']) {
      children.add(TextFormField(
        decoration: InputDecoration(
            border: const UnderlineInputBorder(),
            labelText: SchemaItem.labelNames[key],
            labelStyle: ts),
        // initialValue: widget.json[key],
        controller: controllers[key],
        // onChanged: (value) => updateString(key, value),
        // validator: (value) => ,
        // onFieldSubmitted: (value) => {widget.json[key] = value},
        // onEditingComplete: () => {widget.json[key] = value},
        minLines: 1,
        maxLines: null,
      ));
    }

    for (String key in ['prompt_xy', 'prompt_yx']) {
      children.add(Focus(
        onFocusChange: (bool f) =>
            {if (!f) widget.json[key] = controllers[key]!.text},
        child: TextFormField(
          decoration: InputDecoration(
              border: const UnderlineInputBorder(),
              labelText: SchemaItem.labelNames[key],
              labelStyle: ts),
          // initialValue: widget.json[key],
          controller: controllers[key],
          onFieldSubmitted: (value) => {},
          validator: (value) => validatePrompt(value, key == 'prompt_xy'),
          autovalidateMode: AutovalidateMode.always,
          minLines: 1,
          maxLines: null,
        ),
      ));
    }

    for (String key in ['domain', 'range']) {
      children.add(TextFormField(
        decoration: InputDecoration(
            border: const UnderlineInputBorder(),
            labelText: SchemaItem.labelNames[key],
            labelStyle: ts),
        // initialValue: widget.json[key].join(', '),
        controller: controllers[key],
        onFieldSubmitted: (value) => {widget.json[key] = value},
        validator: validateTypeList,
        // autofillHints: widget.typeHints,
        // auto
      ));
    }

    children.addAll(
      <Widget>[
        for (String key in [
          "reflexive",
          "irreflexive",
          "symmetric",
          "antisymmetric",
          "transitive"
        ])
          CheckboxListTile(
            value: widget.json[key],
            onChanged: (e) => {
              setState(() {
                widget.json[key] = !widget.json[key];
              })
            },
            title: Text(key),
            controlAffinity: ListTileControlAffinity.leading,
            dense: true,
          ),
      ],
    );

    return Column(
      children: children,
    );
  }

  String? validateNotEmpty(String? value) {
    if (value == null || value.isEmpty) {
      return 'Required field.';
    }
    return null;
  }

  String? validateTypeList(String? value) {
    if (validateTypes && widget.typeHints.isNotEmpty && value != null) {
      List<String> pieces = value.split(", ");
      if (!widget.typeHints.containsAll(pieces)) {
        return "Unknown entity type.";
      }
    }
    return null;
  }

  static String? validatePrompt(String? value, bool xy) {
    if (value != null) {
      if (!value.contains('?x')) return "Missing ?x (must be lowercase)";
      if (!value.contains('?y')) return "Missing ?y (must be lowercase)";
      String lc = value[value.length - 1];
      if (!".?!".contains(lc)) return "Prompts should end with punctuation.";
      if (xy) {
        if (value.indexOf("?x") > value.indexOf("?y")) {
          return "?x (${value.indexOf("?x")}) must occur before ?y (${value.indexOf("?y")})";
        }
      } else if (value.indexOf("?y") > value.indexOf("?x")) {
        return "?x must occur before ?y";
      }

      return null;
    }
    return "Please enter a prompt.";
  }
}

// List<SchemaItem> generateItems(int numberOfItems) {
//   return List<SchemaItem>.generate(numberOfItems, (int index) {
//     return SchemaItem(
//       headerValue: 'Panel $index',
//       expandedValue: 'This is item number $index',
//     );
//   });
// }

class SchemaPanelList extends StatefulWidget {
  const SchemaPanelList(
      {super.key,
      required this.schemaJson,
      required this.typeHints,
      required this.onNewSchema});
  final JSONList schemaJson;
  final Set<String> typeHints;
  final void Function(JSONList) onNewSchema;

  @override
  State<SchemaPanelList> createState() => _SchemaPanelListState();
}

class _SchemaPanelListState extends State<SchemaPanelList> {
  // late final List<SchemaItem> _data;
  late final List<ExpansionTileController> _expandies;
  final formKey = GlobalKey<FormState>();
  // bool validateTypes = true;

  @override
  void initState() {
    // _data = [];
    _expandies = [];
    // print("schemaJSON: ${widget.schemaJson}");
    loadSchema(widget.schemaJson);
    super.initState();
  }

  void saveSchema() async {
    if (formKey.currentState!.validate()) {
      String? saveLoc = await saveJSONFile();
      if (saveLoc != null) {
        await File(saveLoc).writeAsString(jsonEncode(<String, dynamic>{
          for (JSONObject v in widget.schemaJson)
            v['rel_id']: v..remove('rel_id')
        }));
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Schema failed to validate, cannot save it.'),
        ),
      );
    }
    // print(_data.map((e) => e.toJSON()).toList());
  }

  void loadSchema(JSONList json) {
    // _data.clear();
    // widget.schemaJson = json;
    _expandies.clear();
    for (JSONObject _ in widget.schemaJson) {
      // _data.add(SchemaItem.fromJson(relJson));
      _expandies.add(ExpansionTileController());
    }
  }

  Future<JSONList> readSchemaFile(XFile? dir) async {
    if (dir != null) {
      try {
        return convertDefaultSchemaJSON(
            jsonDecode(await dir.readAsString()) as JSONObject);
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('The selected file was not a valid JSON List.'),
          ),
        );
        print(e);
      }
    }
    return [];
  }

  JSONList convertDefaultSchemaJSON(JSONObject oldJson) {
    JSONList newJson = [];
    for (String key in oldJson.keys) {
      newJson.add(oldJson[key] as JSONObject..putIfAbsent('rel_id', () => key));
    }
    return newJson;
  }

  void onEditSchema(rel, field, newValue) {}

  void deleteFromSchema(int index) {
    // if (index < _data.length) {
    // _data.removeAt(index);
    if (index < widget.schemaJson.length) {
      widget.schemaJson.removeAt(index);
      _expandies.removeAt(index);
    }
  }

  @override
  void didUpdateWidget(covariant SchemaPanelList oldWidget) {
    if (widget.schemaJson != oldWidget.schemaJson) {
      // loadSchema(_data);
      loadSchema(widget.schemaJson);
    }
    super.didUpdateWidget(oldWidget);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
              ),
              onPressed: () async => widget.onNewSchema(
                  await readSchemaFile(await getJSONFile(context))),
              icon: const Icon(Icons.file_open),
              label: const Text(
                'Load schema from file...',
              ),
            ),
            const SizedBox(width: 12),
            // else //if (selectedSchema != Schema.none)
            ElevatedButton.icon(
              icon: const Icon(Icons.save),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
              ),
              onPressed: saveSchema,
              label: const Text(
                'Save schema to file...',
              ),
            ),
            Expanded(
              child: CheckboxListTile(
                value: validateTypes,
                onChanged: (e) => {
                  setState(() {
                    validateTypes = !validateTypes;
                  })
                },
                title: const Text("Validate types"),
                controlAffinity: ListTileControlAffinity.leading,
              ),
            ),
          ],
        ),
        const Divider(
          thickness: 2.0,
        ),
        Expanded(
          child: SingleChildScrollView(
            scrollDirection: Axis.vertical,
            child: _buildPanel(),
          ),
        ),
        const Divider(
          thickness: 2.0,
        ),
      ],
    );
  }

  Widget _buildPanel() {
    return Form(
      key: formKey,
      child: ListView.builder(
        itemBuilder: (context, index) => ExpansionTile(
          key: Key('s_${widget.schemaJson[index]['name']}@$index'),
          maintainState: false,
          title: Row(
            children: [
              Padding(
                padding: const EdgeInsets.only(right: 8.0),
                child: Checkbox(
                  value: false,
                  onChanged: (e) => {
                    setState(() {
                      // fetchSchema = !fetchSchema;
                    })
                  },
                  visualDensity: const VisualDensity(
                      horizontal: VisualDensity.minimumDensity),
                ),
              ),
              Text(widget.schemaJson[index]['name']),
              IconButton(
                onPressed: () {
                  // ExpansionTileController ex = _expandies[index];
                  setState(() {
                    deleteFromSchema(index);
                    // ex.collapse();
                  });
                },
                icon: const Icon(Icons.delete),
              ),
            ],
          ),
          // controller: _expandies[index],
          children: [
            SizedBox(
              width: MediaQuery.sizeOf(context).width * 0.4,
              child: SchemaItem(
                  // key: Key('e_${widget.schemaJson[index]['name']}@$index'),
                  json: widget.schemaJson[index],
                  typeHints: widget.typeHints),
            ),
          ],
        ),
        itemCount: widget.schemaJson.length,
        shrinkWrap: true,
      ),
    );

    // print(_data.length);
    // return ExpansionPanelList(
    //   // expansionCallback: (int index, bool isExpanded) {
    //   //   setState(() {
    //   //     // _data[index].isExpanded = isExpanded;
    //   //   });
    //   // },
    //   children: _data.map<ExpansionPanel>((SchemaItem item) {
    //     return ExpansionPanel(
    //       headerBuilder: (BuildContext context, bool isExpanded) {
    //         return ListTile(
    //           title: Row(
    //             children: [
    //               Text(isExpanded
    //                   ? item.json['rel_id']
    //                   : '${item.json['rel_id']}: ${item.json['name']}'),
    //               if (isExpanded)
    //                 IconButton(
    //                     onPressed: () {
    //                       setState(() {
    //                         _data.removeWhere((SchemaItem currentItem) =>
    //                             item == currentItem);
    //                       });
    //                     },
    //                     icon: const Icon(Icons.delete))
    //             ],
    //           ),
    //           subtitle: isExpanded ? null : Text(item.json['desc']),
    //         );
    //       },
    //       body: SizedBox(
    //         width: MediaQuery.sizeOf(context).width * 0.4,
    //         child: item,
    //       ),
    //       // isExpanded: item.isExpanded,
    //     );
    //   }).toList(),
    // );
  }
}
