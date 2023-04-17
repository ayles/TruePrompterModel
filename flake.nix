{
  description = "TruePrompterModel";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        rec {
          packages.truepromptermodel = import ./default.nix { inherit pkgs; };
          packages.default = packages.truepromptermodel;
          packages.dockerImage = pkgs.dockerTools.buildImage {
            name = "truepromptermodel";
            tag = "latest";
            copyToRoot = pkgs.buildEnv {
              name = "root";
              paths = [
                packages.truepromptermodel
              ];
              pathsToLink = [ ];
            };
            config =
            {
            };
          };
          devShells.default = (
            pkgs.mkShell.override {
              stdenv = packages.truepromptermodel.stdenv; }
          ) {
            packages = packages.truepromptermodel.nativeBuildInputs;
            buildInputs = packages.truepromptermodel.buildInputs;
            shellHook = packages.truepromptermodel.shellHook;
          };
        }
      );
}

