{
  description = "Nix flake for v-diffusion-pytorch";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
    in {
      packages.${system}.default = pkgs.python3Packages.buildPythonPackage {
        pname = "v-diffusion-pytorch";
        version = "0.1.0";
        format = "setuptools";
        src = ./.;
        build-system = [ pkgs.python3Packages.setuptools ];
        dependencies = with pkgs.python3Packages; [ torch ftfy regex torchvision tqdm ];
        
        pythonRelaxDeps = true;
        doCheck = false;
      };
    };
}
