{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  pythonEnv = pkgs.python39.withPackages (p: with p; [
    datasets
    librosa
    lxml
    soundfile
    torch-bin
    torchaudio-bin
    transformers
  ]);

  venvDir = "./.venv";

  createVenv = ''
    if [ ! -d "${venvDir}" ]; then
      ${pythonEnv.interpreter} -m venv --system-site-packages ${venvDir}
    fi
  '';

  activateVenv = ''
    . ${venvDir}/bin/activate
  '';

  installPythonPackages = ''
    pip install phonemizer evaluate jiwer
  '';

  setupEnvs = ''
    export PHONEMIZER_ESPEAK_LIBRARY=$(find ${pkgs.espeak}/lib -name libespeak-ng.so)
  '';
in
(pkgs.mkShell.override {  }) {
  buildInputs = with pkgs; [
  ];

  packages = with pkgs; [
    (callPackage ./mitlm {})
    espeak
    mbrola
    phonetisaurus
    python39Packages.pip
    python39Packages.virtualenv
    pythonEnv
  ];

  shellHook = createVenv + activateVenv + installPythonPackages + setupEnvs;
}

