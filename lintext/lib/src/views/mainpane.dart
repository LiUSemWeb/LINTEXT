import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:rel_ex_interface/src/util/const.dart';
import 'package:rel_ex_interface/src/util/json.dart';
import 'package:rel_ex_interface/src/views/analyzeview.dart';
import 'package:rel_ex_interface/src/views/relationview.dart';
import 'package:rel_ex_interface/src/views/tokenview.dart';

import 'package:http/http.dart' as http;
// import '../util/dir.dart';

class SharedState {
  List<String> knownEntityTypes = [];
}

/// Displays a list of SampleItems.
class MainPageView extends StatefulWidget {
  const MainPageView({
    super.key,
    required this.settings,
    // this.items = const [SampleItem(1), SampleItem(2), SampleItem(3)],
  });
  final SettingsMemory settings;
  static const routeName = '/';
  @override
  State<MainPageView> createState() => _MainPageViewState();
}

enum DataSet {
  custom('Custom', ['--']),
  docred('DocRED', ['dev', 'train']),
  biored('BioRED', ['train', 'dev', 'test']);

  const DataSet(this.label, this.sets);
  final String label;
  final List<String> sets;
}

enum Schema {
  none('None', ''),
  custom('Custom', ''),
  docred('DocRED', ''),
  docredt10('DocRED top 10', ''),
  biored('BioRED', ''),
  bioredExp('BioRED expanded', ''),
  load('Load: ', '');

  const Schema(this.label, this.dir);
  final String label;
  final String dir;
}

Future<JSONObject> sendRequestToServer(body) async {
  try {
    var response = await http.post(Uri.parse('$serverUrl:$port'),
        body: json.encode(body),
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        });
    return json.decode(response.body);
  } catch (e) {
    print("Error connecting to the server: $e");
  }
  return {};
  // print(tokensJson);
}

class _MainPageViewState extends State<MainPageView>
    with SingleTickerProviderStateMixin {
  final TextEditingController controller = TextEditingController();
  final TextEditingController datasetController = TextEditingController();
  final TextEditingController subsetController = TextEditingController();
  final TextEditingController docnumController =
      TextEditingController(text: "0");
  final TextEditingController schemaController = TextEditingController();
  final TextEditingController searchController = TextEditingController();
  int fromDoc = 1;
  DataSet? selectedDataSet = DataSet.custom;
  Schema? selectedSchema = Schema.none;
  // IconLabel? selectedIcon;
  JSONObject tokensJson = {'tokens': []};
  bool fetchSchema = true;
  bool showTokens = false;
  List<dynamic> rankList = [];

  final ss = SharedState();

  late JSONList schemaJson;

  late TabController tc;

  @override
  void initState() {
    super.initState();
    tc = TabController(length: 2, vsync: this);
    schemaJson = [];
  }

  Future<void> onSave({String? query}) async {
    // List<String> tokens = controller.text.split(' ');
    // Map<String, dynamic> tokensJson = {'tokens': []};

    var body = {'method': '', 'body': {}, 'schema': fetchSchema};

    if (query != null) {
      body['method'] = 'find';
      body['body'] = {
        'query': query,
        'dataset': datasetController.text,
        'subset': subsetController.text,
        'docnum': fromDoc
      };
    } else if (selectedDataSet == DataSet.custom) {
      body['method'] = 'parse';
      body['body'] = {'text': controller.text};
      // var body = json.encode({'text': controller.text});
    } else {
      body['method'] = 'fetch';
      body['body'] = {
        'dataset': datasetController.text,
        'subset': subsetController.text,
        'docnum': docnumController.text
      };
    }
    tokensJson = await sendRequestToServer(body);
    setState(() {
      try {
        if (tokensJson.containsKey('schema')) {
          schemaJson = tokensJson['schema'];
        }
        if(tokensJson.containsKey('docnum')) {
          fromDoc = tokensJson['docnum'] as int;
          docnumController.value = TextEditingValue(text: fromDoc.toString());
        }
        TokensView.resetColors();
        // print("schemaJson: $schemaJson");
      } catch (e) {
        print('Big whoops $e');
      }
    });
  }

  void addNewRelation() {
    setState(() {
      schemaJson.add({
        'rel_id': "",
        'name': "",
        'desc': "",
        'prompt_xy': "?x ?y.",
        'prompt_yx': "?y ?x.",
        'domain': [],
        'range': [],
        'reflexive': false,
        'irreflexive': false,
        'symmetric': false,
        'antisymmetric': false,
        'transitive': false,
        'checked': false
      });
    });
  }

  Future<void> doAnalyze(
      {required JSONList schema, Map<String, Object>? analyzeParams}) async {
    JSONObject formattedSchema = {
      for (JSONObject v in schema)
        if (v['checked']) v['rel_id']: Map.from(v)..remove('rel_id')
    };
    print("DoAnalyze");
    JSONObject r = await sendRequestToServer({
      // text='', schema={}, dataset='', doc=-1, subset='', model=__def_model,
      'method': 'analyze',
      'body': {
        'text': '',
        'schema': formattedSchema,
        'dataset': datasetController.text,
        'subset': subsetController.text,
        'doc': int.parse(docnumController.text),
        // 'model': 'bert-large-uncased',
        // 'num_passes': numPasses,
      }..addAll(analyzeParams ?? {}),
    });
    print("We got $r");
    setState(() {
      rankList = r['results'] ?? [];
    });
  }

  Widget? getFAB() {
    print(tc.index);

    return ListenableBuilder(
        listenable: tc,
        builder: (BuildContext context, Widget? child) {
          return tc.index == 0
              ? SizedBox(
                  height: 32,
                  width: 32,
                  child: FloatingActionButton(
                    backgroundColor: Colors.blue,
                    // tooltip: 'Increment',
                    onPressed: addNewRelation,
                    child: const Icon(
                      Icons.add,
                      color: Colors.white,
                      size: 24,
                    ),
                  ),
                )
              : const SizedBox(width: 0, height: 0);
        });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            const Expanded(
                child: Text('Linköping University\'s Text Exploration Tool')),
            const VerticalDivider(
              width: 4.0,
            ),
            Expanded(
              child: TabBar(
                controller: tc,
                tabs: const <Widget>[Tab(text: "Schema"), Tab(text: "Analyze")],
              ),
            ),
          ],
        ),
      ),
      floatingActionButton: getFAB(),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 4.0),
        child: Row(
          children: [
            Expanded(
              child: Column(
                children: [
                  Row(
                    children: [
                      DropdownMenu<DataSet>(
                        initialSelection: selectedDataSet,
                        controller: datasetController,
                        // requestFocusOnTap is enabled/disabled by platforms when it is null.
                        // On mobile platforms, this is false by default. Setting this to true will
                        // trigger focus request on the text field and virtual keyboard will appear
                        // afterward. On desktop platforms however, this defaults to true.
                        requestFocusOnTap: true,
                        label: const Text('Choose data set'),
                        onSelected: (DataSet? dset) {
                          setState(() {
                            selectedDataSet = dset;
                          });
                        },
                        dropdownMenuEntries: DataSet.values
                            .map<DropdownMenuEntry<DataSet>>((DataSet dset) {
                          return DropdownMenuEntry<DataSet>(
                            value: dset,
                            label: dset.label,
                          );
                        }).toList(),
                      ),
                      if (selectedDataSet != DataSet.custom) ...[
                        const SizedBox(width: 24),
                        DropdownMenu<String>(
                          initialSelection: selectedDataSet?.sets[0],
                          controller: subsetController,
                          // requestFocusOnTap is enabled/disabled by platforms when it is null.
                          // On mobile platforms, this is false by default. Setting this to true will
                          // trigger focus request on the text field and virtual keyboard will appear
                          // afterward. On desktop platforms however, this defaults to true.
                          requestFocusOnTap: true,
                          // label: const Text('Choose data set'),
                          onSelected: (String? dset) {
                            // setState(() {
                            //   selectedDataSet = dset;
                            // });
                          },
                          dropdownMenuEntries: selectedDataSet!.sets
                              .map<DropdownMenuEntry<String>>((String dset) {
                            return DropdownMenuEntry<String>(
                              value: dset,
                              label: dset,
                            );
                          }).toList(),
                        ),
                        const SizedBox(width: 24),
                        SizedBox(
                          width: 60,
                          child: Focus(
                            onFocusChange: (hasFocus) => {
                              if (!hasFocus && docnumController.text.isEmpty)
                                docnumController.text = "0"
                            },
                            canRequestFocus: false,
                            child: TextFormField(
                              maxLines: 1,
                              maxLength: 6,
                              controller: docnumController,
                              keyboardType: TextInputType.number,
                              inputFormatters: <TextInputFormatter>[
                                FilteringTextInputFormatter.digitsOnly
                              ],
                              onChanged: (value) => {
                                if (value.isNotEmpty)
                                  docnumController.text =
                                      int.parse(value).toString()
                              },
                              decoration: const InputDecoration(
                                labelText: "Doc #",
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(
                          width: 24,
                        ),
                        ElevatedButton(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.blue,
                            foregroundColor: Colors.white,
                          ),
                          onPressed: onSave,
                          child: const Text('Fetch'),
                        ),
                        Flexible(
                          child: CheckboxListTile(
                            value: fetchSchema,
                            onChanged: (e) => {
                              setState(() {
                                fetchSchema = !fetchSchema;
                              })
                            },
                            title: const Text(
                              "Fetch Schema",
                            ),
                            controlAffinity: ListTileControlAffinity.leading,
                            dense: true,
                            contentPadding:
                                const EdgeInsets.only(left: 8.0, right: 0),
                            visualDensity: const VisualDensity(
                                horizontal: VisualDensity.minimumDensity),
                          ),
                        ),
                        Flexible(
                          child: CheckboxListTile(
                            value: showTokens,
                            onChanged: (e) => {
                              setState(() {
                                showTokens = !showTokens;
                              })
                            },
                            title: const Text(
                              "Show tokens",
                            ),
                            controlAffinity: ListTileControlAffinity.leading,
                            dense: true,
                            contentPadding:
                                const EdgeInsets.only(left: 8.0, right: 0),
                            visualDensity: const VisualDensity(
                                horizontal: VisualDensity.minimumDensity),
                          ),
                        ),
                      ] else
                        ...[]
                      // Container()
                      // Text('Usghfds')
                      // else Container(),
                    ],
                  ),
                  if (selectedDataSet == DataSet.custom) ...[
                    TextFormField(
                      decoration: const InputDecoration(
                        hintText: 'Text to process',
                        labelText: 'Text:',
                      ),
                      autofocus: false,
                      controller: controller,
                      maxLines: null,
                      keyboardType: TextInputType.text,
                    ),
                    TextButton(
                      onPressed: onSave,
                      child: const Text('Tokenize'),
                    ),
                  ] else ...[
                    const Divider(),
                    Flexible(
                      child: ExpansionTile(
                        title: const Text(
                          'Find Document:',
                          style: TextStyle(fontSize: 14),
                        ),
                        dense: true,
                        children: [
                          Row(
                            children: [
                              Flexible(
                                child: Padding(
                                  padding: const EdgeInsets.symmetric(horizontal: 24.0),
                                  child: TextFormField(
                                    controller: searchController,
                                    decoration:
                                        const InputDecoration(hintText: 'Query'),
                                  ),
                                ),
                              ),
                              ElevatedButton(
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.blue,
                                  foregroundColor: Colors.white,
                                ),
                                onPressed: () => onSave(query: searchController.text),
                                child: const Text('Find...'),
                              ),
                            ],
                          )
                          // Flexible(
                          //   child: Row(
                          //     children: [
                          //       Flexible(
                          //         child: Padding(
                          //           padding:
                          //               const EdgeInsets.symmetric(horizontal: 24.0),
                          //           child: Flexible(
                          //             child: TextFormField(
                          //               decoration:
                          //                   const InputDecoration(hintText: 'Query'),
                          //             ),
                          //           ),
                          //         ),
                          //       ),
                          //       // ElevatedButton(
                          //       //   style: ElevatedButton.styleFrom(
                          //       //     backgroundColor: Colors.blue,
                          //       //     foregroundColor: Colors.white,
                          //       //   ),
                          //       //   onPressed: onSave,
                          //       //   child: const Text('Find...'),
                          //       // ),
                          //     ],
                          //   ),
                          // )
                          // const VerticalDivider(thickness: 32,),
                        ],
                      ),
                    ),
                  ],
                  const Divider(),
                  Wrap(
                      children: TokensView.fromJson(tokensJson,
                          showTokens: showTokens)),
                  const Divider(),
                  const Text("Entity types"),
                  Wrap(children: TokensView.typeListWidgetFromJson(tokensJson)),
                ],
              ),
            ),
            const VerticalDivider(),
            // DefaultTabController(
            //     length: 2,
            //     child: TabBar(tabs: [Tab(text: "Schema"), Tab(text: "Analyze")])),
        
            Expanded(
              child: Column(
                children: [
                  // TabBar(
                  //   controller: tc,
                  //   tabs: [Tab(text: "Schema"), Tab(text: "Analyze")],
                  // ),
                  Expanded(
                    child: TabBarView(
                      controller: tc,
                      children: [
                        SchemaPanelList(
                          schemaJson: schemaJson,
                          typeHints: TokensView.typeListFromJson(tokensJson),
                          onNewSchema: (JSONList newSchema) => {
                            setState(() {
                              schemaJson = newSchema;
                            })
                          },
                        ),
                        AnalyzeView(
                          callback: (
                                  {required Map<String, Object> analyzeParams}) =>
                              doAnalyze(
                                  schema: schemaJson,
                                  analyzeParams: analyzeParams),
                          rankList: rankList,
                          schemaJson: schemaJson,
                          settings: widget.settings,
                        ),
                        // Text('Emptier')
                      ],
                    ),
                  ),
                ],
              ),
            ),
            // SchemaPanelList(
            //   schemaJson: schemaJson,
            //   typeHints: TokensView.typeListFromJson(tokensJson),
            //   onNewSchema: (JSONList newSchema) => {
            //     setState(() {
            //       schemaJson = newSchema;
            //     })
            //   },
            // ),
          ],
        ),
      ),
      // ),
    );
  }
}
