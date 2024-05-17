{
  description = "Neural network optimizer based solely on reducing oscillation.";
  inputs = {
    check-and-compile = {
      inputs = {
        flake-utils.follows = "flake-utils";
        nixfmt.follows = "nixfmt";
        nixpkgs.follows = "nixpkgs";
      };
      url = "github:wrsturgeon/check-and-compile";
    };
    flake-utils.url = "github:numtide/flake-utils";
    nixfmt = {
      inputs.flake-utils.follows = "flake-utils";
      url = "github:serokell/nixfmt";
    };
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs =
    {
      check-and-compile,
      flake-utils,
      nixfmt,
      nixpkgs,
      self,
    }:
    let
      pname = "damper";
      pyname = "damper";
      version = "0.0.1";
      src = ./.;
      jax =
        p: py:
        py.jax.overridePythonAttrs (
          old:
          old
          // {
            doCheck = false;
            propagatedBuildInputs = old.propagatedBuildInputs ++ [ py.jaxlib-bin ];
          }
        );
      default-pkgs =
        p: py:
        with py;
        [
          beartype
          jaxtyping
        ]
        ++ (builtins.map (f: f p py) [
          check-and-compile.lib.with-pkgs
          jax
        ]);
      check-pkgs =
        p: py: with py; [
          hypothesis
          mypy
          pytest
        ];
      ci-pkgs =
        p: py: with py; [
          black
          coverage
        ];
      dev-pkgs =
        p: py: with py; [
          matplotlib
          python-lsp-server
        ];
      lookup-pkg-sets =
        ps: p: py:
        builtins.concatMap (f: f p py) ps;
    in
    {
      lib.with-pkgs =
        pkgs: pypkgs:
        # pkgs.stdenv.mkDerivation {
        #   inherit pname version src;
        #   propagatedBuildInputs = lookup-pkg-sets [ default-pkgs ] pkgs pypkgs;
        #   buildPhase = ":";
        #   installPhase = ''
        #     mkdir -p $out/${pypkgs.python.sitePackages}
        #     mv ./${pyname} $out/${pypkgs.python.sitePackages}/${pyname}
        #   '';
        # };
        pypkgs.buildPythonPackage {
          inherit pname version src;
          pyproject = true;
          build-system = with pypkgs; [ setuptools ];
          dependencies = default-pkgs pkgs pypkgs;
        };
    }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowBroken = true;
        };
        python-version = "312";
        pypkgs = pkgs."python${python-version}Packages";
        python-with = ps: "${pypkgs.python.withPackages (lookup-pkg-sets ps pkgs)}/bin/python";
      in
      {
        apps.ci = {
          type = "app";
          program = "${
            let
              pname = "ci";
              python = python-with [
                default-pkgs
                check-pkgs
                ci-pkgs
              ];
              find = "${pkgs.findutils}/bin/find";
              nixfmt-bin = "${nixfmt.packages.${system}.default}/bin/nixfmt";
              rm = "${pkgs.coreutils}/bin/rm";
              xargs = "${pkgs.findutils}/bin/xargs";
              exec = ''
                #!${pkgs.bash}/bin/bash

                set -eu

                export JAX_ENABLE_X64=1

                ${rm} -fr result
                ${find} . -name '*.nix' | ${xargs} ${nixfmt-bin} --check
                ${python} -m black --check .
                ${python} -m mypy .

                ${python} -m coverage run --omit='/nix/*' -m pytest -Werror test.py
                ${python} -m coverage report -m --fail-under=100
              '';
            in
            pkgs.stdenv.mkDerivation {
              inherit pname version src;
              buildPhase = ":";
              installPhase = ''
                mkdir -p $out/bin
                echo "${exec}" > $out/bin/${pname}
                chmod +x $out/bin/${pname}
              '';
            }
          }/bin/ci";
        };
        devShells.default = pkgs.mkShell {
          JAX_ENABLE_X64 = "1";
          packages = (
            lookup-pkg-sets [
              default-pkgs
              check-pkgs
              ci-pkgs
              dev-pkgs
            ] pkgs pypkgs
          );
        };
        packages.default = self.lib.with-pkgs pkgs pypkgs;
      }
    );
}
