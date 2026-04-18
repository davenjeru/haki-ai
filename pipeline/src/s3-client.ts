import { S3Client } from "@aws-sdk/client-s3";
import { AWS_REGION } from "./config.js";

/**
 * Returns an S3Client pointed at LocalStack when AWS_ENDPOINT_URL is set,
 * or real AWS otherwise. Set AWS_ENDPOINT_URL=http://localhost:4566 to use LocalStack.
 */
export function makeS3Client(): S3Client {
  const endpoint = process.env.AWS_ENDPOINT_URL;
  if (endpoint) {
    return new S3Client({
      region: AWS_REGION,
      endpoint,
      forcePathStyle: true, // required for LocalStack path-style bucket URLs
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID ?? "test",
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY ?? "test",
      },
    });
  }
  return new S3Client({ region: AWS_REGION });
}

export const s3 = makeS3Client();
