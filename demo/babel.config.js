module.exports = {
    presets: [
      ['@babel/preset-env', {targets: {node: 'current'}, modules: false}],
      ['@babel/preset-react', {targets: {node: 'current'}}], // add this
    //   ['@babel/transform-runtime', 'babel-plugin-transform-import-meta']
    ]
  };