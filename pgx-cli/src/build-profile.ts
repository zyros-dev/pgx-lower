import type { ResolvedProfileConfig } from "./project-config.js";

export function cmakeArgs(profile: ResolvedProfileConfig): string[] {
  return Object.entries(profile.build.cmake.args)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `-D${key}=${formatCmakeValue(value)}`);
}

export function renderConfigureCommand(profile: ResolvedProfileConfig, sourceDir: string): string[] {
  const command = ["cmake", "-S", sourceDir, "-B", profile.build.build_dir];
  if (profile.build.cmake.generator) {
    command.push("-G", profile.build.cmake.generator);
  }
  command.push(...cmakeArgs(profile));
  return command;
}

export function renderBuildExplain(profileName: string, profile: ResolvedProfileConfig): string {
  const lines = [
    `profile: ${profileName}`,
    `build_dir: ${profile.build.build_dir}`,
    `cmake.generator: ${profile.build.cmake.generator}`,
    ...Object.entries(profile.build.cmake.args)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, value]) => `cmake.${key}: ${String(value)}`),
    ...Object.entries(profile.runtime)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, value]) => `runtime.${key}: ${String(value)}`)
  ];
  return `${lines.join("\n")}\n`;
}

function formatCmakeValue(value: string | number | boolean): string {
  if (value === true) return "ON";
  if (value === false) return "OFF";
  return String(value);
}
