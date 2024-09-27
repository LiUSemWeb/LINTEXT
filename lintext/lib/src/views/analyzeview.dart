import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:rel_ex_interface/src/util/json.dart';

class AnalyzeView extends StatefulWidget {
  const AnalyzeView(
      {super.key,
      required this.callback,
      required this.rankList,
      required this.schemaJson});

  final AsyncCallback callback;
  final List<dynamic> rankList;
  final JSONList schemaJson;

  @override
  State<AnalyzeView> createState() => _AnalyzeViewState();
}

class _AnalyzeViewState extends State<AnalyzeView> {
  @override
  Widget build(BuildContext context) {
    int checkCount =
        widget.schemaJson.where((e) => e['checked'] ?? false).length;

    return Column(
      children: [
        Row(
          children: [
            Wrap(
              children: [
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    foregroundColor: Colors.white,
                  ),
                  onPressed: widget.callback,
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
                  widget.rankList[index][7]
                      ? const Icon(
                          Icons.check_box,
                          color: Colors.blue,
                        )
                      : const Icon(Icons.disabled_by_default_rounded,
                          color: Colors.red),
                  Expanded(child: Text('${widget.rankList[index][6]}')),
                  Padding(
                    padding: const EdgeInsets.only(right: 16.0),
                    child:
                        Text('${widget.rankList[index][8].toStringAsFixed(2)}'),
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
