# Django AI SDK - Documentation

This is the documentation site for Django AI SDK, built with [Hugo](https://gohugo.io/) and the [Hextra](https://github.com/imfing/hextra) theme.

## Local Development

Pre-requisites: [Hugo](https://gohugo.io/getting-started/installing/) and [Go](https://golang.org/doc/install)

```shell
cd docs
hugo mod tidy
hugo server --logLevel debug --disableFastRender -p 1313
```

### Update theme

```shell
hugo mod get -u
hugo mod tidy
```
