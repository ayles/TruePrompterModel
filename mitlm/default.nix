{ pkgs ? import <nixpkgs> {} }:

with pkgs;

stdenv.mkDerivation rec {
  pname = "mitlm";
  version = "0.4.2";

  src = fetchFromGitHub {
    owner = "mitlm";
    repo = pname;
    rev = "v${version}";
    sha256 = "sha256-gmTv2wRJU8YecyhZeb8IoBtIFfXgPewtAC/vW/mE2hI=";
  };

  nativeBuildInputs = [ autoreconfHook autoconf-archive gfortran ];
  buildInputs = [ zlib ];

  meta = with lib; {
    description = "The MIT Language Modeling Toolkit";
    homepage = "https://github.com/mitlm/mitlm";
    license = licenses.gpl2Plus;
    platforms = platforms.unix;
    maintainers = [ ];
  };
}

