{ pkgs ? import <nixpkgs> {} }:

with pkgs;

stdenv.mkDerivation rec {
  pname = "flite";
  version = "2.2";

  src = fetchFromGitHub {
    owner = "festvox";
    repo = "flite";
    rev = "v${version}";
    sha256 = "1n0p81jzndzc1rzgm66kw9ls189ricy5v1ps11y0p2fk1p56kbjf";
  };

  buildInputs = lib.optionals stdenv.isLinux [ alsa-lib ];

  patches = [
    (fetchpatch {
      url = "https://github.com/festvox/flite/commit/54c65164840777326bbb83517568e38a128122ef.patch";
      sha256 = "sha256-hvKzdX7adiqd9D+9DbnfNdqEULg1Hhqe1xElYxNM1B8=";
    })
  ];

  configureFlags = [
    "--enable-shared"
  ] ++ lib.optionals stdenv.isLinux [ "--with-audio=alsa" ];

  installPhase = ''
    make install
    cd testsuite && make lex_lookup && cp lex_lookup $out/bin && cd ..
  '';

  enableParallelBuilding = false;

  meta = with lib; {
    description = "A small, fast run-time speech synthesis engine";
    homepage = "http://www.festvox.org/flite/";
    license = licenses.bsdOriginal;
    platforms = platforms.all;
  };
}

