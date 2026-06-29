"""插件发现(M4)—— 第三方包经 entry points 自动注册。

第三方包只要在自己的 pyproject.toml 里声明:

    [project.entry-points."rein.plugins"]
    my_provider = "my_pkg.providers:MyProvider"

`load_plugins()` 就能在运行时把它们发现并加载出来 —— 无需用户手动 import。
用标准库 `importlib.metadata`,零额外依赖。

约定:加载结果是 {名字: 对象},对象是什么(Provider 类 / Runtime 类 / 中间件 / 注册回调)
由插件作者决定;rein 只负责「发现 + load」,不强加结构(保持极薄)。
"""

GROUP = "rein.plugins"


def load_plugins(group: str = GROUP) -> dict:
    """发现并加载某个 entry points group 下的全部插件,返回 {名字: 已加载对象}。

    单个插件加载失败不影响其它(记下错误对象,不抛)—— 一个坏插件不该拖垮整个进程。
    """
    from importlib.metadata import entry_points

    # Python 3.10+ 的 entry_points(group=...) 接口
    try:
        eps = entry_points(group=group)
    except TypeError:  # 兼容极老的 importlib.metadata 行为
        eps = entry_points().get(group, [])  # type: ignore[arg-type]

    loaded: dict = {}
    for ep in eps:
        try:
            loaded[ep.name] = ep.load()
        except Exception as e:  # 坏插件:记录异常对象,继续加载其它
            loaded[ep.name] = e
    return loaded


def plugin_names(group: str = GROUP) -> list[str]:
    """只列出可发现的插件名(不加载),用于排查 / 展示。"""
    from importlib.metadata import entry_points

    try:
        eps = entry_points(group=group)
    except TypeError:
        eps = entry_points().get(group, [])  # type: ignore[arg-type]
    return [ep.name for ep in eps]
