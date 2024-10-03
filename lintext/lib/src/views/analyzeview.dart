import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:rel_ex_interface/src/util/json.dart';
import 'package:rel_ex_interface/src/views/statementpane.dart';

typedef AnalyzeCallback = Future<void> Function({required Map<String, Object> analyzeParams});

class AnalyzeView extends StatefulWidget {
  const AnalyzeView(
      {super.key,
      required this.callback,
      required this.rankList,
      required this.schemaJson});

  final AnalyzeCallback callback;
  final List<dynamic> rankList;
  final JSONList schemaJson;

  @override
  State<AnalyzeView> createState() => _AnalyzeViewState();
}

class _AnalyzeViewState extends State<AnalyzeView> {
  TextEditingController numPassesController = TextEditingController(text: "0");

  Future<void> callback() async {
    await widget.callback(
      analyzeParams: {
        'num_passes': int.parse(numPassesController.text),
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
            SizedBox(
              width: 60,
              child: Focus(
                onFocusChange: (hasFocus) => {
                  if (!hasFocus && numPassesController.text.isEmpty)
                    numPassesController.text = "0"
                },
                canRequestFocus: false,
                child: TextFormField(
                  maxLines: 1,
                  maxLength: 1,
                  controller: numPassesController,
                  keyboardType: TextInputType.number,
                  inputFormatters: <TextInputFormatter>[
                    FilteringTextInputFormatter.digitsOnly
                  ],
                  onChanged: (value) => {
                    if (value.isNotEmpty)
                      numPassesController.text = int.parse(value).toString()
                  },
                  decoration: const InputDecoration(
                    labelText: "# Passes",
                  ),
                ),
              ),
            ),
            const VerticalDivider(thickness: 32,),
          ],
        ),
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
