{ pkgs }:

let
  pythonEnv = pkgs.python310.withPackages (p: with p; [
    datasets
    librosa
    lxml
    soundfile
    tkinter
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
    pip install phonemizer evaluate jiwer matplotlib panphon epitran
  '';

  setupEnvs = ''
    export PHONEMIZER_ESPEAK_LIBRARY=$(find ${pkgs.espeak}/lib -name libespeak-ng.so)
  '';
in
(pkgs.mkShell.override {  }) {
  buildInputs = with pkgs; [
  ];

  nativeBuildInputs = with pkgs; [
    (callPackage ./flite {})
    (callPackage ./mitlm {})
    espeak
    mbrola
    phonetisaurus
    python310Packages.pip
    python310Packages.virtualenv
    pythonEnv
  ];

  shellHook = createVenv + activateVenv + installPythonPackages + setupEnvs;
}

