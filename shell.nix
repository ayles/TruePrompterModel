{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  pythonEnv = pkgs.python39.withPackages (p: with p; [
    lxml
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
    pip install ipapy datasets librosa soundfile phonemizer
  '';
in
(pkgs.mkShell.override {  }) {
  buildInputs = with pkgs; [
  ];

  packages = with pkgs; [
    (callPackage ./mitlm {})
    phonetisaurus
    python39Packages.pip
    python39Packages.virtualenv
    pythonEnv
  ];

  shellHook = createVenv + activateVenv + installPythonPackages;
}

