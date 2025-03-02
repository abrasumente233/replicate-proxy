{
  description = "Replicate API Proxy";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          packages.default = pkgs.buildGoModule {
            pname = "replicate-proxy";
            version = "0.1.0";
            src = ./.;
            # vendorSha256 = null; # Set to null for zero dependencies, or the correct hash if you have dependencies
            vendorHash = null;
          };

          devShells.default = pkgs.mkShell {
            buildInputs = with pkgs; [
              go
              gopls
              gotools
              go-tools
            ];
          };
        }) // {
      overlays.default = _: prev: {
        rob = self.packages.${prev.system}.default;
      };
      nixosModules.default = import ./nixos-module.nix self;
    };
}
