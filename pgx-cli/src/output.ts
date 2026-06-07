export type StreamSample = {
  head: string;
  tail: string;
  truncated: boolean;
  omittedBytes?: number;
  omittedLines?: number;
};

export function streamSampleToText(sample: StreamSample | string): string {
  if (typeof sample === "string") {
    return sample;
  }
  if (!sample.truncated) {
    return sample.head + sample.tail;
  }
  const omitted = sample.omittedLines !== undefined
    ? `lines: ${sample.omittedLines}`
    : `bytes: ${sample.omittedBytes ?? 0}`;
  return `${sample.head}[... omitted ${omitted}; full transcript in run artifact ...]\n${sample.tail}`;
}
