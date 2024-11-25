String port = '13679';
String serverUrl = 'http://localhost';

String defaultModel = 'bert-large-cased';

typedef SettingsMemory = Map<String, Object>;
SettingsMemory newDefaultSettingsMemory() {
  return {
    "num_passes": "0",
    "model": defaultModel,
    "scorer": "pll",
  };
}
