self:
{ pkgs, config, lib, ... }:
with lib;
let
  cfg = config.services.replicate-proxy;
in
{
  options.services.replicate-proxy = {
    enable = mkEnableOption "replicate-proxy";
    package = lib.mkOption {
      description = "replicate-proxy package to use";
      type = lib.types.package;
      default = self.packages.${pkgs.system}.default;
    };
    port = lib.mkOption {
      description = "port to listen on";
      type = lib.types.port;
      default = 9876;
    };
  };
  config = mkIf cfg.enable {
    systemd.services.replicate-proxy = {
      description = "Replicate Proxy";
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/proxy --port ${toString cfg.port}";
        Restart = "always";
        RestartSec = "30";
        DynamicUser = true;
      };
    };
  };
}
