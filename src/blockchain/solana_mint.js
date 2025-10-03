const { createUmi } = require('@metaplex-foundation/umi-bundle-defaults');
const { createFungible, mplTokenMetadata } = require('@metaplex-foundation/mpl-token-metadata');
const { generateSigner } = require('@metaplex-foundation/umi');

async function mintQHNFT(name, color, seed, fqc) {
  const umi = createUmi('https://api.mainnet-beta.solana.com').use(mplTokenMetadata());
  const mintSigner = generateSigner(umi);
  const metadata = {
    name: `${name}'s Cosmic Relic`,
    symbol: 'QHNFT',
    uri: `ipfs://QmEggMetadata/${name}_${fqc}.json`,
    sellerFeeBasisPoints: 500,
  };
  await createFungible(umi, {
    mint: mintSigner,
    name: metadata.name,
    symbol: metadata.symbol,
    uri: metadata.uri,
    sellerFeeBasisPoints: metadata.sellerFeeBasisPoints,
    metadata: { fqc, seed, color },
  }).sendAndConfirm(umi);
  return { mint: mintSigner.publicKey.toString(), tx: 'mock_tx' };
}

module.exports = { mintQHNFT };
