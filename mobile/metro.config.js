/** Metro config — allow .txt ticker lists as assets */
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.assetExts = [...config.resolver.assetExts, 'txt'];

module.exports = config;
