import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as opensearchserverless from 'aws-cdk-lib/aws-opensearchserverless';
import * as iam from 'aws-cdk-lib/aws-iam';

export class RagBedrockOpensearchStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ✅ 1. Bucket S3 pour stocker les PDFs
    const ragBucket = new s3.Bucket(this, 'RagDocumentsBucket', {
      bucketName: `rag-documents-${this.account}`,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    });

    // ✅ 2. Encryption Policy OpenSearch (obligatoire)
    const encryptionPolicy = new opensearchserverless.CfnSecurityPolicy(this, 'EncryptionPolicy', {
      name: 'rag-encryption-policy',
      type: 'encryption',
      policy: JSON.stringify({
        Rules: [{ ResourceType: 'collection', Resource: ['collection/rag-collection'] }],
        AWSOwnedKey: true,
      }),
    });

    // ✅ 3. Network Policy OpenSearch (obligatoire)
    const networkPolicy = new opensearchserverless.CfnSecurityPolicy(this, 'NetworkPolicy', {
      name: 'rag-network-policy',
      type: 'network',
      policy: JSON.stringify([{
        Rules: [
          { ResourceType: 'collection', Resource: ['collection/rag-collection'] },
          { ResourceType: 'dashboard', Resource: ['collection/rag-collection'] },
        ],
        AllowFromPublic: true,
      }]),
    });

    // ✅ 4. Collection OpenSearch Serverless
    const collection = new opensearchserverless.CfnCollection(this, 'RagCollection', {
      name: 'rag-collection',
      type: 'VECTORSEARCH',
    });
    collection.addDependency(encryptionPolicy);
    collection.addDependency(networkPolicy);

    // ✅ 5. Rôle IAM pour Bedrock + OpenSearch + S3
    const ragRole = new iam.Role(this, 'RagExecutionRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      roleName: 'rag-execution-role',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonBedrockFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonOpenSearchServiceFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
      ],
    });

    // ✅ 6. Outputs
    new cdk.CfnOutput(this, 'BucketName', {
      value: ragBucket.bucketName,
      description: 'Nom du bucket S3',
    });
    new cdk.CfnOutput(this, 'CollectionEndpoint', {
      value: collection.attrCollectionEndpoint,
      description: 'Endpoint OpenSearch Serverless',
    });
    new cdk.CfnOutput(this, 'RoleArn', {
      value: ragRole.roleArn,
      description: 'ARN du rôle IAM',
    });
  }
}