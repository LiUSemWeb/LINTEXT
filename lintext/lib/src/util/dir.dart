import 'package:flutter/material.dart';

import 'package:file_selector/file_selector.dart';

Future<String?> getFile(BuildContext context) async {
  const String confirmButtonText = 'Open Schema File';
  final String? directoryPath = await getDirectoryPath(
    confirmButtonText: confirmButtonText,
  );
  return directoryPath;

  // if (directoryPath == null) {
  //   // Operation was canceled by the user.
  //   return;
  // }
  // if (context.mounted) {
  //   await showDialog<void>(
  //     context: context,
  //     builder: (BuildContext context) => TextDisplay(directoryPath),
  //   );
  // }
}

const XTypeGroup jsonType =
    XTypeGroup(label: 'JSON', extensions: <String>['json', 'txt']);

Future<XFile?> getJSONFile(BuildContext context) async {
  const String confirmButtonText = 'Open Schema File';
  final XFile? file = await openFile(
    acceptedTypeGroups: [jsonType],
    confirmButtonText: confirmButtonText,
  );
  return file;
}

Future<String?> saveJSONFile() async {
  FileSaveLocation? outFilePath = await getSaveLocation(
    acceptedTypeGroups: [jsonType],
    suggestedName: "string_info.json",
    confirmButtonText: "Save",
  );
  return outFilePath?.path;
}

/// Widget that displays a text file in a dialog
class TextDisplay extends StatelessWidget {
  /// Default Constructor
  const TextDisplay(this.directoryPath, {super.key});

  /// Directory path
  final String directoryPath;

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Selected Directory'),
      content: Scrollbar(
        child: SingleChildScrollView(
          child: Text(directoryPath),
        ),
      ),
      actions: <Widget>[
        TextButton(
          child: const Text('Close'),
          onPressed: () => Navigator.pop(context),
        ),
      ],
    );
  }
}
