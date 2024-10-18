import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:rel_ex_interface/src/util/const.dart';
import 'package:rel_ex_interface/src/util/json.dart';
import 'package:rel_ex_interface/src/views/statementpane.dart';

typedef AnalyzeCallback = Future<void> Function(
    {required Map<String, Object> analyzeParams});

enum Scorer {
  pll('PLL', 'pll'),
  csd('Cosine', 'csd'),
  jsd('Jensen-Shannon', 'csd'),
  hsd('Hellinger', 'csd'),
  msd('Mean Square', 'msd'),
  esd('Euclidean', 'esd');

  const Scorer(this.name, this.tag);
  final String name;
  final String tag;
}

class AnalyzeView extends StatefulWidget {
  const AnalyzeView(
      {super.key,
      required this.callback,
      required this.rankList,
      required this.schemaJson,
      required this.settings});

  final AnalyzeCallback callback;
  final List<dynamic> rankList;
  final JSONList schemaJson;
  final SettingsMemory settings;

  @override
  State<AnalyzeView> createState() => _AnalyzeViewState();
}

class _AnalyzeViewState extends State<AnalyzeView> {
  Scorer? selectedScorer = Scorer.pll;

  late Map<String, TextEditingController> controllers;
  List<String> controlledSettings = ["num_passes", "model", "scorer"];

  @override
  void initState() {
    controllers = {};
    for (String setting in controlledSettings) {
      controllers[setting] =
          TextEditingController(text: widget.settings[setting] as String);
      controllers[setting]!.addListener(() {
        widget.settings[setting] = controllers[setting]!.text;
      });
    }
    super.initState();
  }

  Future<void> callback() async {
    print(controllers['scorer']!.text);
    await widget.callback(
      analyzeParams: {
        'num_passes': int.parse(controllers['num_passes']!.text),
        'model': controllers['model']!.text,
        'scorer': selectedScorer?.tag ?? 'pll',
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    int checkCount =
        widget.schemaJson.where((e) => e['checked'] ?? false).length;

    return Column(
      children: [
        ExpansionTile(
          title: Row(
            key: const Key('AnalyzeOptions'),
            children: [
              Wrap(
                children: [
                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      foregroundColor: Colors.white,
                    ),
                    onPressed: callback,
                    child: const Text('Analyze!'),
                  ),
                ],
              ),
              Padding(
                padding: const EdgeInsets.only(left: 8.0),
                child: Align(
                  alignment: Alignment.center,
                  child: Text(
                    '$checkCount/${widget.schemaJson.length} selected.',
                    style: (checkCount == 0)
                        ? const TextStyle(color: Colors.red)
                        : null,
                  ),
                ),
              ),
            ],
          ),
          children: [
            Row(
              children: [
                SizedBox(
                  width: 60,
                  child: Focus(
                    onFocusChange: (hasFocus) {
                      if (!hasFocus) {
                        TextEditingController npc = controllers['num_passes']!;
                        if (npc.text.isEmpty) {
                          npc.text = "0";
                        } else if (int.parse(npc.text) > 10) {
                          npc.text = "10";
                        }
                      }
                    },
                    canRequestFocus: false,
                    child: TextFormField(
                      maxLines: 1,
                      // maxLength: 1,
                      controller: controllers['num_passes'],
                      keyboardType: TextInputType.number,
                      inputFormatters: <TextInputFormatter>[
                        FilteringTextInputFormatter.digitsOnly
                      ],
                      onChanged: (value) => {
                        if (value.isNotEmpty)
                          controllers['num_passes']!.text =
                              int.parse(value).toString()
                      },
                      decoration: const InputDecoration(
                        labelText: "# Passes",
                      ),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(left: 8.0),
                  child: SizedBox(
                    width: 180,
                    child: Focus(
                      onFocusChange: (hasFocus) => {
                        if (!hasFocus && controllers['model']!.text.isEmpty)
                          controllers['model']!.text = defaultModel
                      },
                      canRequestFocus: false,
                      child: TextFormField(
                        maxLines: 1,
                        // maxLength: 256,
                        controller: controllers['model'],
                        decoration: const InputDecoration(
                          labelText: "Model name",
                        ),
                      ),
                    ),
                  ),
                ),
                DropdownMenu<Scorer>(
                  initialSelection: selectedScorer,
                  controller: controllers['scorer']!,
                  // requestFocusOnTap is enabled/disabled by platforms when it is null.
                  // On mobile platforms, this is false by default. Setting this to true will
                  // trigger focus request on the text field and virtual keyboard will appear
                  // afterward. On desktop platforms however, this defaults to true.
                  requestFocusOnTap: true,
                  label: const Text('Choose data set'),
                  // onSelected: (Scorer? dset) {
                  //   setState(() {
                  //     widget.settings["scorer"] = dset;
                  //   });
                  // },
                  dropdownMenuEntries: Scorer.values
                      .map<DropdownMenuEntry<Scorer>>((Scorer scorer) {
                    return DropdownMenuEntry<Scorer>(
                      value: scorer,
                      label: scorer.name,
                    );
                  }).toList(),
                ),
              ],
            ),
            // const VerticalDivider(thickness: 32,),
          ],
        ),
        const IntrinsicWidth(child: VerticalDivider()),
        // Expanded(
        //   child: SingleChildScrollView(
        //     child: Text(
        //       widget.rankList.join("\n"),
        //     ),
        //   ),
        // ),
        Expanded(
          child: SingleChildScrollView(
            child: ListView.builder(
              itemBuilder: (context, index) => Row(
                // controller: _expandies[index],
                children: [
                  widget.rankList[index][5]
                      ? const Icon(
                          Icons.check_box,
                          color: Colors.blue,
                        )
                      : const Icon(Icons.disabled_by_default_rounded,
                          color: Colors.red),
                  // Expanded(child: Text('${widget.rankList[index][4]}')),
                  Expanded(
                    child: StatementView(
                      text: widget.rankList[index][4].split(' '),
                      scores: (widget.rankList[index][6]),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(right: 16.0),
                    child: Text(
                        '${widget.rankList[index][widget.rankList[index].length - 1].toStringAsFixed(2)}'),
                  ),
                  // SizedBox(
                  //   width: MediaQuery.sizeOf(context).width * 0.4,
                  //   child: SchemaItem(
                  //       // key: Key('e_${widget.schemaJson[index]['name']}@$index'),
                  //       json: widget.schemaJson[index],
                  //       typeHints: widget.typeHints),
                  // ),
                ],
              ),
              itemCount: widget.rankList.length,
              shrinkWrap: true,
            ),
          ),
        ),
      ],
    );
  }
}
